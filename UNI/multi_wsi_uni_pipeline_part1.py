# multi_wsi_uni_pipeline_part1.py
import os
import sys
import csv
import time
import glob
import shutil
import signal
import tempfile
import subprocess
from multiprocessing import Process
from typing import List, Tuple, Optional

import numpy as np
import torch
import openslide
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoModel

# --------------- Globals set in init_worker ---------------
MODEL = None
DEVICE = None
PREPROCESS = None
BATCH_SIZE = 32
CHECKPOINT_EVERY = 10

# --------------- Utilities ---------------
def preferred_tmp_root(user_tmp: Optional[str] = None) -> str:
    """
    Choose a temp root that exists and has space.
    Order: provided -> $SLURM_TMPDIR -> /scratch -> /scratch3 -> /tmp
    """
    for p in [
        user_tmp,
        os.environ.get("SLURM_TMPDIR"),
        f"/scratch/{os.environ.get('USER','')}",
        f"/scratch3/{os.environ.get('USER','')}",
        "/tmp",
    ]:
        if p and os.path.isdir(p):
            return p
    return "/tmp"

def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)

def run_cmd(cmd: List[str], cwd: Optional[str] = None) -> None:
    """Run a shell command, raise on non-zero."""
    r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")

# --------------- Model init per worker ---------------
def init_worker(gpu_id=None, model_name="MahmoodLab/UNI", batch_size=32, checkpoint_every=10):
    global MODEL, DEVICE, PREPROCESS, BATCH_SIZE, CHECKPOINT_EVERY
    BATCH_SIZE = int(batch_size)
    CHECKPOINT_EVERY = int(checkpoint_every)

    if gpu_id is None:
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    print(f"[worker init] loading {model_name} on {DEVICE} (gpu_id={gpu_id}) ...", flush=True)
    MODEL = AutoModel.from_pretrained(model_name)
    MODEL.eval()
    MODEL.to(DEVICE)
    for p in MODEL.parameters():
        p.requires_grad = False

    PREPROCESS = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    print("[worker init] done.", flush=True)

# --------------- Patch enumeration / loading ---------------
def enumerate_patches_for_wsi(wsi_path, patch_size=224, step_size=224, white_thresh=0.8):
    slide = openslide.OpenSlide(wsi_path)
    w, h = slide.dimensions
    coords = []
    # inclusive end guard
    for y in range(0, max(0, h - patch_size + 1), step_size):
        for x in range(0, max(0, w - patch_size + 1), step_size):
            region = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
            arr = np.array(region, copy=False)
            white_frac = np.mean(np.all(arr > 220, axis=-1))
            if white_frac <= white_thresh:
                coords.append((x, y))
    slide.close()
    return coords

def load_patch_batch(wsi_path, coords_batch, patch_size=224):
    slide = openslide.OpenSlide(wsi_path)
    tensors = []
    for (x, y) in coords_batch:
        patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
        t = PREPROCESS(patch)  # (3,224,224)
        tensors.append(t)
    slide.close()
    return torch.stack(tensors, dim=0)  # (B,3,224,224)

# --------------- Core per-WSI processing ---------------
def process_single_wsi(wsi_path, output_dir,
                       patch_size=224, step_size=224,
                       batch_size=None, white_thresh=0.8):
    global MODEL, DEVICE, BATCH_SIZE, CHECKPOINT_EVERY
    if MODEL is None:
        raise RuntimeError("Worker MODEL not initialized. Call init_worker first.")

    if batch_size is None:
        batch_size = BATCH_SIZE

    os.makedirs(output_dir, exist_ok=True)
    slide_id = os.path.splitext(os.path.basename(wsi_path))[0]
    final_path = os.path.join(output_dir, f"{slide_id}.npz")
    tmp_path   = os.path.join(output_dir, f".{slide_id}.tmp.npz")

    if os.path.exists(final_path):
        print(f"[skip] already processed: {slide_id}")
        return final_path

    coords = enumerate_patches_for_wsi(wsi_path, patch_size=patch_size, step_size=step_size, white_thresh=white_thresh)
    total = len(coords)
    if total == 0:
        print(f"[no patches] {slide_id}")
        np.savez_compressed(final_path,
                            embeddings=np.zeros((0, MODEL.config.hidden_size), dtype=np.float32),
                            metadata=np.empty((0, 3), dtype=np.int32),
                            slide_id=slide_id)
        return final_path

    # resume?
    if os.path.exists(tmp_path):
        loaded = np.load(tmp_path, allow_pickle=True)
        embeddings_so_far = [loaded["embeddings"]]
        meta_so_far = list(loaded["metadata"])
        start_idx = len(meta_so_far)
        print(f"[resume] {slide_id} from {start_idx}/{total}")
    else:
        embeddings_so_far, meta_so_far, start_idx = [], [], 0

    t0 = time.time()
    for start in range(start_idx, total, batch_size):
        batch_coords = coords[start:start+batch_size]
        batch_t = load_patch_batch(wsi_path, batch_coords, patch_size=patch_size).to(DEVICE)

        with torch.no_grad():
            try:
                out = MODEL(batch_t)
            except TypeError:
                out = MODEL(pixel_values=batch_t)
            emb = out.last_hidden_state[:, 0, :].float().cpu().numpy()

        embeddings_so_far.append(emb)
        for i, (x, y) in enumerate(batch_coords):
            meta_so_far.append((x, y, start + i))

        # checkpoint?
        batches_completed = (start // batch_size) + 1
        if (batches_completed % CHECKPOINT_EVERY == 0) or (start + batch_size >= total):
            emb_arr = np.vstack(embeddings_so_far) if embeddings_so_far else np.zeros((0, MODEL.config.hidden_size), dtype=np.float32)
            meta_arr = np.array(meta_so_far, dtype=np.int32)
            np.savez_compressed(tmp_path, embeddings=emb_arr, metadata=meta_arr, slide_id=slide_id)

            done = len(meta_so_far)
            dt = max(1e-6, time.time() - t0)
            pps = done / dt
            eta_min = (total - done) / pps / 60.0 if pps > 0 else float("inf")
            print(f"[{slide_id}] {done}/{total} patches, {pps:.1f} p/s, ETA {eta_min:.1f} min (checkpoint)")

    # finalize
    if os.path.exists(tmp_path):
        os.replace(tmp_path, final_path)
    else:
        emb_arr = np.vstack(embeddings_so_far) if embeddings_so_far else np.zeros((0, MODEL.config.hidden_size), dtype=np.float32)
        meta_arr = np.array(meta_so_far, dtype=np.int32)
        np.savez_compressed(final_path, embeddings=emb_arr, metadata=meta_arr, slide_id=slide_id)

    print(f"[done] saved {final_path}")
    return final_path

# --------------- GDC manifest parsing & download ---------------
def parse_manifest(manifest_path: str) -> List[Tuple[str, str]]:
    """
    Returns list of (file_id, filename) for .svs entries from gdc_manifest_part_001.txt (TSV).
    Expected columns: id filename md5 size state
    """
    out = []
    with open(manifest_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            fid = row.get("id") or row.get("file_id") or row.get("file_id".upper())
            fname = row.get("filename") or row.get("file_name") or row.get("file_name".upper())
            if not fid or not fname:
                continue
            if fname.lower().endswith((".svs", ".tif", ".ndpi", ".mrxs")):
                out.append((fid, fname))
    if not out:
        raise RuntimeError(f"No WSI entries (.svs/.tif/.ndpi/.mrxs) found in manifest: {manifest_path}")
    return out

def download_wsi_to_tmp(gdc_client: str, file_id: str, tmp_root: str) -> str:
    """
    Use gdc-client to download a single file_id into a fresh temp dir.
    Returns the path to the downloaded slide file (.svs/.tif/...).
    Directory is created under tmp_root and should be deleted by caller.
    """
    workdir = tempfile.mkdtemp(prefix=f"gdc_{file_id}_", dir=tmp_root)
    # gdc-client download <uuid> -d <dir>
    cmd = [gdc_client, "download", file_id, "-d", workdir]
    run_cmd(cmd, cwd=workdir)

    # gdc-client creates a subfolder named with UUID; search for slide
    slide_candidates = []
    for ext in ("*.svs", "*.tif", "*.ndpi", "*.mrxs"):
        slide_candidates.extend(glob.glob(os.path.join(workdir, "**", ext), recursive=True))
    if not slide_candidates:
        raise RuntimeError(f"GDC download finished but no slide file found for {file_id} in {workdir}")
    # if multiple, take the largest (likely the WSI)
    slide_candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return workdir, slide_candidates[0]

# --------------- Worker main (download -> process -> cleanup) ---------------
def process_one_manifest_entry(entry: Tuple[str, str],
                               output_dir: str,
                               gdc_client: str,
                               tmp_root: str,
                               patch_size: int,
                               step_size: int,
                               batch_size: int,
                               white_thresh: float):
    file_id, filename = entry
    slide_stub = os.path.splitext(filename)[0]
    final_path = os.path.join(output_dir, f"{slide_stub}.npz")
    if os.path.exists(final_path):
        print(f"[skip] already processed (exists): {slide_stub}")
        return

    tmp_dir = None
    try:
        tmp_dir_root = preferred_tmp_root(tmp_root)
        print(f"[download] {slide_stub} -> temp root: {tmp_dir_root}")
        tmp_dir, slide_path = download_wsi_to_tmp(gdc_client, file_id, tmp_dir_root)
        print(f"[process] {slide_stub} from {slide_path}")
        process_single_wsi(slide_path, output_dir,
                           patch_size=patch_size, step_size=step_size,
                           batch_size=batch_size, white_thresh=white_thresh)
    finally:
        if tmp_dir and os.path.isdir(tmp_dir):
            try:
                shutil.rmtree(tmp_dir)
                print(f"[cleanup] removed temp dir {tmp_dir}")
            except Exception as e:
                print(f"[warn] failed to remove {tmp_dir}: {e}")

# --------------- Parallel driver (manifest mode) ---------------
def run_manifest_parallel(manifest_path: str,
                          output_dir: str,
                          gdc_client_path: str,
                          num_workers: int = 2,
                          gpu_ids: Optional[List[int]] = None,
                          patch_size: int = 224,
                          step_size: int = 224,
                          batch_size: int = 32,
                          white_thresh: float = 0.8,
                          checkpoint_every: int = 10,
                          tmp_root: Optional[str] = None):
    entries = parse_manifest(manifest_path)
    print(f"[manifest] {len(entries)} WSI slide entries found")

    # build per-worker settings
    if gpu_ids is None or len(gpu_ids) == 0:
        worker_cfg = [(None, "MahmoodLab/UNI", batch_size, checkpoint_every)] * num_workers
    else:
        worker_cfg = []
        for i in range(num_workers):
            worker_cfg.append((gpu_ids[i % len(gpu_ids)], "MahmoodLab/UNI", batch_size, checkpoint_every))

    # split entries to workers round-robin
    buckets = [entries[i::num_workers] for i in range(num_workers)]
    procs: List[Process] = []

    def _worker(entries_chunk, gpu, model_name, bs, ck):
        # init model on this process/GPU
        init_worker(gpu_id=gpu, model_name=model_name, batch_size=bs, checkpoint_every=ck)
        safe_mkdir(output_dir)
        for e in entries_chunk:
            try:
                process_one_manifest_entry(e, output_dir, gdc_client_path, tmp_root,
                                           patch_size, step_size, bs, white_thresh)
            except Exception as ex:
                print(f"[error] {e}: {ex}", flush=True)

    print(f"[parallel] workers={num_workers}, gpu_ids={gpu_ids}")
    for wi in range(num_workers):
        if not buckets[wi]:
            continue
        gpu, model_name, bs, ck = worker_cfg[wi]
        p = Process(target=_worker, args=(buckets[wi], gpu, model_name, bs, ck))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

# --------------- CLI ---------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="UNI WSI feature extraction (GDC manifest streaming).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", type=str, help="Path to gdc_manifest_part_001.txt (TSV).")
    group.add_argument("--wsi-dir", type=str, help="(Legacy) Local WSI directory mode.")
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--gdc-client", type=str,
                        default="/users/lesego/projects/histgen_dataset/gdc-client",
                        help="Path to gdc-client binary.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gpu-ids", type=str, default="", help="e.g. '0,1' or '' for CPU")
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--step-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--white-thresh", type=float, default=0.8)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--tmp-root", type=str, default=None, help="Preferred temp root. Defaults to $SLURM_TMPDIR or /tmp.")
    args = parser.parse_args()

    gpu_ids = None
    if args.gpu_ids.strip():
        gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip() != ""]

    if args.manifest:
        run_manifest_parallel(
            manifest_path=args.manifest,
            output_dir=args.out_dir,
            gdc_client_path=args.gdc_client,
            num_workers=args.num_workers,
            gpu_ids=gpu_ids,
            patch_size=args.patch_size,
            step_size=args.step_size,
            batch_size=args.batch_size,
            white_thresh=args.white_thresh,
            checkpoint_every=args.checkpoint_every,
            tmp_root=args.tmp_root,
        )
    else:
        # Legacy local-dir mode (if you ever want it)
        # Reuse your earlier run_parallel for local WSIs if needed:
        from multiprocessing import Process as _P  # placeholder
        raise SystemExit("Local --wsi-dir mode is disabled in this build. Use --manifest instead.")

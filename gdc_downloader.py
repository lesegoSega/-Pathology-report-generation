#!/usr/bin/env python3
"""
GDC Manifest WSI Downloader - Minimal Dependencies Version
Only uses Python standard library + requests

Usage:
    python gdc_downloader_simple.py --manifest manifest.txt --output ./wsi_data --max-files 5
"""

import os
import sys
import argparse
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Optional

try:
    import requests
except ImportError:
    print("❌ Error: 'requests' module is required")
    print("Install with: pip install requests")
    sys.exit(1)


class GDCDownloader:
    """Download WSI files from GDC using manifest file"""
    
    def __init__(self, output_dir: str = "./wsi_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.base_url = "https://api.gdc.cancer.gov/data"
        
        self.wsi_dir = self.output_dir / "WSIs"
        self.wsi_dir.mkdir(exist_ok=True)
        
        self.log_file = self.output_dir / "download_log.txt"
        self.failed_log = self.output_dir / "failed_downloads.txt"
    
    def parse_manifest(self, manifest_file: str) -> List[Dict]:
        """Parse GDC manifest file (TSV format)"""
        print(f"📋 Parsing manifest file: {manifest_file}")
        
        try:
            manifest_data = []
            
            with open(manifest_file, 'r') as f:
                # Read header
                header = f.readline().strip().split('\t')
                
                # Expected columns: id, filename, md5, size, state
                if 'id' not in header or 'filename' not in header:
                    print("❌ Invalid manifest format")
                    print("Expected columns: id, filename, md5, size, state")
                    sys.exit(1)
                
                # Get column indices
                id_idx = header.index('id')
                filename_idx = header.index('filename')
                md5_idx = header.index('md5')
                size_idx = header.index('size')
                
                # Read data rows
                for line in f:
                    if line.strip():
                        parts = line.strip().split('\t')
                        if len(parts) >= 4:
                            manifest_data.append({
                                'id': parts[id_idx],
                                'filename': parts[filename_idx],
                                'md5': parts[md5_idx],
                                'size': int(parts[size_idx])
                            })
            
            print(f"✅ Found {len(manifest_data)} files in manifest")
            total_size = sum(item['size'] for item in manifest_data) / (1024**3)
            print(f"📊 Total download size: {total_size:.2f} GB")
            
            return manifest_data
            
        except Exception as e:
            print(f"❌ Error parsing manifest: {e}")
            sys.exit(1)
    
    def calculate_md5(self, filepath: Path) -> str:
        """Calculate MD5 checksum"""
        md5_hash = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    
    def verify_download(self, filepath: Path, expected_md5: str) -> bool:
        """Verify file integrity"""
        if not filepath.exists():
            return False
        
        print(f"  🔍 Verifying MD5...")
        actual_md5 = self.calculate_md5(filepath)
        
        if actual_md5 == expected_md5:
            print(f"  ✅ MD5 verified")
            return True
        else:
            print(f"  ❌ MD5 mismatch!")
            return False
    
    def download_file(self, file_id: str, filename: str, expected_md5: str, 
                     file_size: int, max_retries: int = 3) -> bool:
        """Download a single file"""
        output_path = self.wsi_dir / filename
        
        # Check if already downloaded
        if output_path.exists() and output_path.stat().st_size == file_size:
            print(f"  ✅ File already exists, skipping...")
            return True
        
        url = f"{self.base_url}/{file_id}"
        
        # Attempt download with retries
        for attempt in range(1, max_retries + 1):
            try:
                print(f"  📥 Downloading (attempt {attempt}/{max_retries})...")
                
                response = requests.get(url, stream=True, timeout=300)
                response.raise_for_status()
                
                # Download in chunks
                downloaded = 0
                chunk_size = 8192
                
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Print progress every 10MB
                            if downloaded % (10 * 1024 * 1024) == 0:
                                progress_mb = downloaded / (1024 * 1024)
                                total_mb = file_size / (1024 * 1024)
                                print(f"  📊 Progress: {progress_mb:.1f} MB / {total_mb:.1f} MB")
                
                print(f"  ✅ Download complete ({downloaded / (1024**2):.1f} MB)")
                
                # Verify
                if self.verify_download(output_path, expected_md5):
                    self._log_success(file_id, filename)
                    return True
                else:
                    output_path.unlink()
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
                if output_path.exists():
                    output_path.unlink()
                
                if attempt < max_retries:
                    wait = 2 ** attempt
                    print(f"  ⏳ Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    self._log_failure(file_id, filename, str(e))
                    return False
        
        return False
    
    def download_from_manifest(self, manifest_file: str, max_files: Optional[int] = None,
                              start_from: int = 0):
        """Download files from manifest"""
        manifest_data = self.parse_manifest(manifest_file)
        
        # Apply limits
        if max_files:
            manifest_data = manifest_data[start_from:start_from + max_files]
        else:
            manifest_data = manifest_data[start_from:]
        
        print(f"\n🚀 Starting download of {len(manifest_data)} files...")
        print(f"💾 Output: {self.wsi_dir}")
        print("=" * 60)
        
        successful = 0
        failed = 0
        start_time = time.time()
        
        for i, item in enumerate(manifest_data, start=start_from + 1):
            print(f"\n[{i}/{len(manifest_data) + start_from}] {item['filename']}")
            print(f"  UUID: {item['id']}")
            print(f"  Size: {item['size'] / (1024**2):.1f} MB")
            
            if self.download_file(item['id'], item['filename'], item['md5'], item['size']):
                successful += 1
            else:
                failed += 1
            
            time.sleep(1)  # Rate limiting
        
        # Summary
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("📊 DOWNLOAD SUMMARY")
        print("=" * 60)
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"⏱️  Time: {elapsed / 60:.1f} minutes")
        print(f"📁 Location: {self.wsi_dir}")
        
        if failed > 0:
            print(f"⚠️  Check: {self.failed_log}")
    
    def _log_success(self, file_id: str, filename: str):
        with open(self.log_file, 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{file_id}\t{filename}\tOK\n")
    
    def _log_failure(self, file_id: str, filename: str, error: str):
        with open(self.failed_log, 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{file_id}\t{filename}\t{error}\n")


def main():
    parser = argparse.ArgumentParser(description='Download WSI files from GDC')
    parser.add_argument('--manifest', '-m', required=True, help='GDC manifest file')
    parser.add_argument('--output', '-o', default='./wsi_data', help='Output directory')
    parser.add_argument('--max-files', type=int, help='Max files to download')
    parser.add_argument('--start-from', type=int, default=0, help='Resume from index')
    
    args = parser.parse_args()
    
    if not Path(args.manifest).exists():
        print(f"❌ Manifest not found: {args.manifest}")
        sys.exit(1)
    
    print("🔬 GDC WSI Downloader")
    print("=" * 60)
    
    downloader = GDCDownloader(output_dir=args.output)
    
    try:
        downloader.download_from_manifest(
            manifest_file=args.manifest,
            max_files=args.max_files,
            start_from=args.start_from
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted - resume with --start-from")


if __name__ == "__main__":
    main()
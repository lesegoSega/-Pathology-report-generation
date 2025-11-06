#!/usr/bin/env python3
import csv, os
from pathlib import Path

OUT = "/users/lesego/projects/histgen_dataset/organ_terms.csv"

# (organ, term, weight)
rows = [
    # --- breast ---
    ("breast","invasive ductal carcinoma",6),
    ("breast","Nottingham grade",5),
    ("breast","sentinel lymph node",5),
    ("breast","margin distance",4),
    ("breast","lymphovascular invasion",5),
    ("breast","perineural invasion",3),
    ("breast","ductal carcinoma in situ",4),
    ("breast","HER2 IHC or FISH",4),
    ("breast","ER PR status",4),

    # --- colon/rectum ---
    ("colon/rectum","pericolic fat",6),
    ("colon/rectum","pT3",5),
    ("colon/rectum","tumor budding",4),
    ("colon/rectum","MMR MSI",5),
    ("colon/rectum","CRM margin",4),
    ("colon/rectum","lymphovascular invasion",5),
    ("colon/rectum","perineural invasion",4),
    ("colon/rectum","examined positive nodes",5),

    # --- rectum specific ---
    ("rectum","CRM margin",6),
    ("rectum","mesorectal fascia",5),
    ("rectum","ypT ypN",4),
    ("rectum","MMR MSI",5),
    ("rectum","lymphovascular invasion",5),
    ("rectum","perineural invasion",4),
    ("rectum","distal margin mm",4),

    # --- stomach ---
    ("stomach","Lauren type",5),
    ("stomach","serosal invasion",6),
    ("stomach","greater omentum",5),
    ("stomach","HER2",4),
    ("stomach","examined positive nodes",5),
    ("stomach","proximal distal margins",4),
    ("stomach","perineural invasion",4),

    # --- oral cavity ---
    ("oral cavity","depth of invasion",6),
    ("oral cavity","extranodal extension",5),
    ("oral cavity","perineural invasion",5),
    ("oral cavity","lymphovascular invasion",4),
    ("oral cavity","margin distance",5),
    ("oral cavity","levels II III IV",4),
    ("oral cavity","pT pN",4),

    # --- lung ---
    ("lung","visceral pleural invasion",6),
    ("lung","PD-L1",5),
    ("lung","STAS",5),
    ("lung","lymphovascular invasion",5),
    ("lung","bronchial vascular margins",4),
    ("lung","stations examined",4),
    ("lung","adenocarcinoma pattern",4),
    ("lung","pT pN",4),

    # --- brain ---
    ("brain","IDH mutation",6),
    ("brain","1p19q codeletion",6),
    ("brain","integrated diagnosis",6),
    ("brain","MIB-1 index",5),
    ("brain","necrosis MVP",4),
    ("brain","extent of resection",4),

    # --- uterus (corpus) ---
    ("uterus (corpus)","myometrial invasion",6),
    ("uterus (corpus)","LVSI",6),
    ("uterus (corpus)","FIGO grade",5),
    ("uterus (corpus)","MMR MSI",6),
    ("uterus (corpus)","POLE p53",5),
    ("uterus (corpus)","cervical stromal involvement",5),
    ("uterus (corpus)","adnexa serosa",4),
    ("uterus (corpus)","examined positive nodes",4),

    # --- lymph node ---
    ("lymph node","DLBCL",6),
    ("lymph node","Ki-67",5),
    ("lymph node","BCL2 BCL6 MYC",5),
    ("lymph node","EBV ISH",4),
    ("lymph node","CD20 CD3 panel",5),
    ("lymph node","Lugano stage",4),

    # === NEW ORGANS ===

    # --- skin ---
    ("skin","Breslow depth mm",7),
    ("skin","ulceration",6),
    ("skin","mitotic rate per mm2",6),
    ("skin","Clark level",4),
    ("skin","regression",3),
    ("skin","perineural invasion",4),
    ("skin","lymphovascular invasion",4),
    ("skin","sentinel lymph node",5),
    ("skin","margin distance",4),
    ("skin","melanoma subtype",4),

    # --- liver ---
    ("liver","hepatocellular carcinoma",6),
    ("liver","Edmondson Steiner grade",5),
    ("liver","microvascular invasion",6),
    ("liver","portal vein invasion",5),
    ("liver","tumor thrombus",5),
    ("liver","cirrhosis background",4),
    ("liver","resection margin mm",4),
    ("liver","glypican-3",3),
    ("liver","Arginase-1",3),

    # --- kidney ---
    ("kidney","clear cell RCC",6),
    ("kidney","ISUP grade",5),
    ("kidney","perinephric fat invasion",6),
    ("kidney","renal sinus invasion",5),
    ("kidney","renal vein invasion",5),
    ("kidney","margin status",4),
    ("kidney","adrenal involvement",3),
    ("kidney","pT pN",4),

    # --- pancreas ---
    ("pancreas","pancreatic ductal adenocarcinoma",6),
    ("pancreas","perineural invasion",6),
    ("pancreas","uncinate retroperitoneal margin",6),
    ("pancreas","SMV portal vein margin",5),
    ("pancreas","examined positive nodes",5),
    ("pancreas","tumor size cm",4),
    ("pancreas","R1 resection",4),
    ("pancreas","pT pN",4),

    # --- gallbladder + alias 'gallblader' ---
    ("gallbladder","cholecystectomy",6),
    ("gallbladder","perimuscular connective tissue invasion",6),
    ("gallbladder","cystic duct margin",5),
    ("gallbladder","hepatic bed margin",5),
    ("gallbladder","gallstones",3),
    ("gallbladder","dysplasia",3),
    ("gallbladder","lymphovascular invasion",4),
    ("gallbladder","T2a T2b",4),

    ("gallblader","cholecystectomy",6),
    ("gallblader","perimuscular connective tissue invasion",6),
    ("gallblader","cystic duct margin",5),
    ("gallblader","hepatic bed margin",5),
    ("gallblader","gallstones",3),
    ("gallblader","dysplasia",3),
    ("gallblader","lymphovascular invasion",4),
    ("gallblader","T2a T2b",4),

    # --- prostate ---
    ("prostate","Gleason score",7),
    ("prostate","Grade Group",6),
    ("prostate","perineural invasion",5),
    ("prostate","extraprostatic extension",6),
    ("prostate","seminal vesicle invasion",6),
    ("prostate","margin status",5),
    ("prostate","lymph node status",4),
    ("prostate","tertiary pattern",3),

    # --- thyroid ---
    ("thyroid","papillary thyroid carcinoma",6),
    ("thyroid","follicular variant",5),
    ("thyroid","capsular invasion",6),
    ("thyroid","vascular invasion",6),
    ("thyroid","extrathyroidal extension",6),
    ("thyroid","margin status",4),
    ("thyroid","lymph node metastasis",4),
    ("thyroid","BRAF V600E",3),

    # --- larynx ---
    ("larynx","supraglottic glottic subglottic",6),
    ("larynx","cartilage invasion",6),
    ("larynx","margin status",5),
    ("larynx","perineural invasion",5),
    ("larynx","examined positive nodes",5),
    ("larynx","extranodal extension",5),
    ("larynx","tumor size cm",4),
    ("larynx","pT pN",4),

    # --- pleura ---
    ("pleura","epithelioid mesothelioma",7),
    ("pleura","calretinin",6),
    ("pleura","WT1",5),
    ("pleura","CK5 6",4),
    ("pleura","BAP1 loss",6),
    ("pleura","CDKN2A p16 deletion",5),
    ("pleura","parietal visceral pleura",4),
    ("pleura","pleural invasion",5),

    # --- retroperitoneum ---
    ("retroperitoneum","liposarcoma",6),
    ("retroperitoneum","leiomyosarcoma",6),
    ("retroperitoneum","MDM2 amplification",6),
    ("retroperitoneum","margin status R0 R1",5),
    ("retroperitoneum","adjacent organ involvement",5),
    ("retroperitoneum","tumor size cm",4),
    ("retroperitoneum","necrosis percent",4),

    # --- eye + ocular alias ---
    ("eye","choroidal melanoma",7),
    ("eye","basal diameter mm",6),
    ("eye","thickness mm",6),
    ("eye","ciliary body involvement",5),
    ("eye","scleral invasion",5),
    ("eye","extraocular extension",6),
    ("eye","optic nerve margin",5),
    ("eye","BAP1 status",4),

    ("ocular","choroidal melanoma",7),
    ("ocular","basal diameter mm",6),
    ("ocular","thickness mm",6),
    ("ocular","ciliary body involvement",5),
    ("ocular","scleral invasion",5),
    ("ocular","extraocular extension",6),
    ("ocular","optic nerve margin",5),
    ("ocular","BAP1 status",4),

    # --- tongue (explicit) ---
    ("tongue","depth of invasion mm",7),
    ("tongue","margin distance mm",6),
    ("tongue","perineural invasion",6),
    ("tongue","lymphovascular invasion",5),
    ("tongue","levels II III IV",5),
    ("tongue","extranodal extension",6),
    ("tongue","p16 HPV",4),
    ("tongue","tumor size cm",4),

    # --- ovary ---
    ("ovary","high grade serous carcinoma",7),
    ("ovary","STIC lesion",6),
    ("ovary","fallopian tube fimbriae",5),
    ("ovary","omentum involvement",6),
    ("ovary","peritoneal implants",6),
    ("ovary","FIGO stage",6),
    ("ovary","BRCA status",4),
    ("ovary","residual disease R0",5),

    # --- maxillary sinus ---
    ("maxillary sinus","sinonasal squamous cell carcinoma",6),
    ("maxillary sinus","bone invasion",6),
    ("maxillary sinus","perineural invasion",5),
    ("maxillary sinus","orbital extension",5),
    ("maxillary sinus","margin status mm",5),
    ("maxillary sinus","pterygopalatine fossa",4),
    ("maxillary sinus","inverted papilloma",4),
]

def write_csv(rows, out_path):
    # Deduplicate (organ, term) keeping highest weight
    best = {}
    for organ, term, weight in rows:
        key = (organ.strip(), term.strip())
        if key not in best or weight > best[key]:
            best[key] = weight
    # Sort for readability
    items = sorted(((o,t,w) for (o,t),w in best.items()),
                   key=lambda x: (x[0].lower(), -x[2], x[1].lower()))
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["organ","term","weight"])
        for o,t,wgt in items:
            w.writerow([o,t,wgt])

if __name__ == "__main__":
    write_csv(rows, OUT)
    print(f"Wrote {OUT}")

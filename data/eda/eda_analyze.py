# -*- coding: utf-8 -*-
"""
READ-ONLY analysis of image_inventory.csv.
Outputs:
  - patient_summary.csv
  - anomalies.csv
  - eda_stats.txt   (human-readable dump used to build the report)
  - eda_figures/*.png
Also computes a brightness pass (2nd read) for near-black/near-white detection.
"""
import os, sys, json, re, math
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ROOT, "eda_figures")
os.makedirs(FIG_DIR, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True  # for brightness pass only

INV = os.path.join(ROOT, "image_inventory.csv")
df = pd.read_csv(INV)

out = []
def p(*a):
    s = " ".join(str(x) for x in a)
    out.append(s)
    print(s)

p("############ EDA ANALYSIS ############")
p("total rows:", len(df))

# ---------- 1. Inventory table ----------
p("\n==== 1. INVENTORY (split|class|image_count|non_image|corrupted) ====")
inv_rows = []
for split in ["train","val","test"]:
    for cls in ["NORMAL","PNEUMONIA"]:
        sub = df[(df["split"]==split)&(df["class"]==cls)]
        img = (sub["status"]=="ok").sum()
        nonimg = (sub["status"]=="non_image").sum()
        corr = (sub["status"]=="corrupted").sum()
        inv_rows.append((split,cls,int(img),int(nonimg),int(corr),len(sub)))
        p(f"{split:6s} {cls:10s} images={img:5d} non_image={nonimg} corrupted={corr} total={len(sub)}")
p("Extensions present:", dict(df["extension"].value_counts()))

# ---------- 2. Modes / channels ----------
p("\n==== 2. IMAGE MODES / CHANNELS ====")
ok = df[df["status"]=="ok"].copy()
p("Overall mode counts:", dict(ok["mode"].value_counts()))
p("Overall channels counts:", dict(ok["channels"].value_counts()))
for split in ["train","val","test"]:
    for cls in ["NORMAL","PNEUMONIA"]:
        sub = ok[(ok["split"]==split)&(ok["class"]==cls)]
        p(f"{split:6s} {cls:10s} modes={dict(sub['mode'].value_counts())}")
# RGB verification
rgb = ok[ok["mode"]=="RGB"]
p("Total RGB images:", len(rgb))
p("RGB by class:", dict(rgb["class"].value_counts()))
p("RGB by split/class:", dict(rgb.groupby(['split','class']).size()))
p("All RGB are PNEUMONIA?", bool((rgb["class"]=="PNEUMONIA").all()))
p("Sample RGB filenames:", list(rgb["filename"].head(10)))

# ---------- 3. Dimensions ----------
p("\n==== 3. IMAGE DIMENSIONS ====")
ok["width"]=ok["width"].astype(float)
ok["height"]=ok["height"].astype(float)
ok["aspect_ratio"]=ok["aspect_ratio"].astype(float)
def stats_block(sub, label):
    if len(sub)==0:
        p(label, "EMPTY"); return
    w=sub["width"]; h=sub["height"]; ar=sub["aspect_ratio"]
    q=[0.05,0.25,0.5,0.75,0.95]
    p(f"--- {label} (n={len(sub)}) ---")
    p(f"  width : min={w.min():.0f} max={w.max():.0f} mean={w.mean():.1f} median={w.median():.0f}")
    p(f"  height: min={h.min():.0f} max={h.max():.0f} mean={h.mean():.1f} median={h.median():.0f}")
    p(f"  aspect: min={ar.min():.3f} max={ar.max():.3f} mean={ar.mean():.3f} median={ar.median():.3f}")
    p(f"  width  quantiles {q}: {[round(w.quantile(x),1) for x in q]}")
    p(f"  height quantiles {q}: {[round(h.quantile(x),1) for x in q]}")
    p(f"  aspect quantiles {q}: {[round(ar.quantile(x),3) for x in q]}")
for split in ["train","val","test"]:
    for cls in ["NORMAL","PNEUMONIA"]:
        stats_block(ok[(ok["split"]==split)&(ok["class"]==cls)], f"{split}/{cls}")
# Aggregate NORMAL vs PNEUMONIA (all splits)
stats_block(ok[ok["class"]=="NORMAL"], "ALL NORMAL")
stats_block(ok[ok["class"]=="PNEUMONIA"], "ALL PNEUMONIA")

# ---------- Figures: histograms ----------
def hist_compare(col, title, fname, bins=60):
    plt.figure(figsize=(8,5))
    n = ok[ok["class"]=="NORMAL"][col].dropna()
    pn = ok[ok["class"]=="PNEUMONIA"][col].dropna()
    rng=(min(n.min(),pn.min()), max(n.max(),pn.max()))
    plt.hist(n, bins=bins, range=rng, alpha=0.5, label=f"NORMAL (n={len(n)})", density=True)
    plt.hist(pn, bins=bins, range=rng, alpha=0.5, label=f"PNEUMONIA (n={len(pn)})", density=True)
    plt.xlabel(col); plt.ylabel("density"); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,fname), dpi=110); plt.close()
    p("figure saved:", fname)
hist_compare("width","Width distribution: NORMAL vs PNEUMONIA","hist_width.png")
hist_compare("height","Height distribution: NORMAL vs PNEUMONIA","hist_height.png")
hist_compare("aspect_ratio","Aspect ratio distribution: NORMAL vs PNEUMONIA","hist_aspect.png", bins=60)

# scatter w vs h
plt.figure(figsize=(7,7))
for cls,c in [("NORMAL","tab:blue"),("PNEUMONIA","tab:orange")]:
    sub=ok[ok["class"]==cls]
    plt.scatter(sub["width"],sub["height"],s=4,alpha=0.25,label=cls,color=c)
plt.xlabel("width"); plt.ylabel("height"); plt.title("Width vs Height"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,"scatter_wh.png"),dpi=110); plt.close()
p("figure saved: scatter_wh.png")

# class bar
plt.figure(figsize=(7,5))
ct = df.groupby(["split","class"]).size().unstack(fill_value=0)
ct = ct.reindex(["train","val","test"])
ct.plot(kind="bar", ax=plt.gca())
plt.title("Class counts per split"); plt.ylabel("count"); plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"class_counts.png"),dpi=110); plt.close()
p("figure saved: class_counts.png")

# ---------- 4. Patient ID analysis ----------
p("\n==== 4. PATIENT ID ANALYSIS ====")
# PNEUMONIA
pne = ok[ok["class"]=="PNEUMONIA"].copy()
p("-- PNEUMONIA --")
p("rows with method 'person_regex':", int((pne['patient_id_method']=='person_regex').sum()))
p("rows with UNKNOWN person prefix:", int((pne['patient_id_method']=='no_person_prefix').sum()))
uk = pne[pne['patient_id_method']=='no_person_prefix']
p("  UNKNOWN examples:", list(uk['filename'].head(10)))
pne_pid = pne[pne['patient_id_candidate']!='UNKNOWN']
uniq = pne_pid['patient_id_candidate'].nunique()
per = pne_pid.groupby('patient_id_candidate').size()
p("unique person IDs:", uniq)
p("images per person: min", int(per.min()), "max", int(per.max()), "median", float(per.median()), "mean", round(float(per.mean()),2))
# bacteria vs virus
pne["subtype"] = pne["filename"].str.extract(r"person\d+_(bacteria|virus)", flags=re.IGNORECASE)[0].str.lower()
p("PNEUMONIA subtype counts:", dict(pne["subtype"].value_counts(dropna=False)))
# cross-split for PNEUMONIA persons
pid_split = pne_pid.groupby('patient_id_candidate')['split'].nunique()
cross = pid_split[pid_split>1]
p("PNEUMONIA person IDs appearing in >1 split:", len(cross))
if len(cross):
    for pid in list(cross.index)[:50]:
        splits = sorted(pne_pid[pne_pid['patient_id_candidate']==pid]['split'].unique())
        p("   CROSS", pid, splits)

# NORMAL naming pattern analysis
p("-- NORMAL --")
nor = ok[ok["class"]=="NORMAL"].copy()
def prefix_of(fn):
    stem=os.path.splitext(fn)[0]
    # token before first digit-group pattern
    return re.sub(r"\d+","#",stem)
nor["pattern"]=nor["filename"].apply(prefix_of)
p("Top NORMAL filename patterns (digits->#):")
for pat,cnt in nor["pattern"].value_counts().head(20).items():
    ex = nor[nor["pattern"]==pat]["filename"].iloc[0]
    p(f"   {cnt:5d}  {pat}   e.g. {ex}")
# method distribution
p("NORMAL patient_id_method counts:", dict(nor["patient_id_method"].value_counts()))
nor_grp = nor.groupby('patient_id_candidate').size()
p("NORMAL group candidates: unique", nor["patient_id_candidate"].nunique(),
  "images/group min", int(nor_grp.min()), "max", int(nor_grp.max()), "median", float(nor_grp.median()))
multi = nor_grp[nor_grp>1].sort_values(ascending=False)
p("NORMAL groups with >1 image (top 20):")
for g,c in multi.head(20).items():
    p(f"   {c:3d}  {g}")
# NORMAL cross-split
nor_split = nor.groupby('patient_id_candidate')['split'].nunique()
nor_cross = nor_split[nor_split>1]
p("NORMAL group candidates appearing in >1 split:", len(nor_cross))
for g in list(nor_cross.index)[:50]:
    splits=sorted(nor[nor['patient_id_candidate']==g]['split'].unique())
    p("   NCROSS", g, splits)

# ---------- patient_summary.csv ----------
psum = ok.groupby(['patient_id_candidate','class','split']).size().reset_index(name='num_images')
psum.to_csv(os.path.join(ROOT,"patient_summary.csv"), index=False, encoding="utf-8")
p("patient_summary.csv written rows:", len(psum))

# ---------- 5. Duplicate / leakage ----------
p("\n==== 5. DUPLICATE / LEAKAGE (sha256) ====")
h = ok[ok["hash"].astype(str).str.len()>0].copy()
dups = h.groupby("hash").filter(lambda g: len(g)>1)
groups = dups.groupby("hash")
p("hashes with duplicates:", dups["hash"].nunique(), "files involved:", len(dups))
within_split=0; cross_split=0; cross_class=0
examples=[]
for hsh,g in groups:
    splits=set(g["split"]); classes=set(g["class"])
    if len(splits)>1: cross_split+=1
    else: within_split+=1
    if len(classes)>1: cross_class+=1
    if len(examples)<40:
        examples.append((hsh[:12], list(zip(g["split"],g["class"],g["filename"]))))
p("duplicate groups within same split:", within_split)
p("duplicate groups spanning >1 split (LEAKAGE):", cross_split)
p("duplicate groups spanning >1 class:", cross_class)
for hsh,items in examples[:40]:
    p("   DUPGROUP", hsh, items)

# ---------- 6. Class imbalance ----------
p("\n==== 6. CLASS IMBALANCE ====")
for split in ["train","val","test"]:
    sub=ok[ok["split"]==split]
    n=(sub["class"]=="NORMAL").sum(); pn=(sub["class"]=="PNEUMONIA").sum()
    ratio = pn/n if n else float('inf')
    p(f"{split:6s} NORMAL={n} PNEUMONIA={pn} ratio PNEU:NORMAL={ratio:.2f}:1")
tv=ok[ok["split"].isin(["train","val"])]
n=(tv["class"]=="NORMAL").sum(); pn=(tv["class"]=="PNEUMONIA").sum()
p(f"train+val NORMAL={n} PNEUMONIA={pn} ratio={pn/n:.2f}:1")

# ---------- 7. Anomalies ----------
p("\n==== 7. SUSPICIOUS / EXTREME IMAGES ====")
anom=[]
def add_anom(row, reason):
    anom.append({"path":row["path"],"split":row["split"],"class":row["class"],
                 "filename":row["filename"],"width":row["width"],"height":row["height"],
                 "aspect_ratio":row["aspect_ratio"],"reason":reason})
ar=ok["aspect_ratio"]
w=ok["width"]; h=ok["height"]
ar_lo, ar_hi = ar.quantile(0.005), ar.quantile(0.995)
p(f"aspect 0.5%/99.5% = {ar_lo:.3f}/{ar_hi:.3f}")
for _,r in ok.iterrows():
    reasons=[]
    if r["aspect_ratio"]<=0.60: reasons.append(f"very_tall_AR={r['aspect_ratio']:.3f}")
    if r["aspect_ratio"]>=1.90: reasons.append(f"very_wide_AR={r['aspect_ratio']:.3f}")
    if r["width"]<400 or r["height"]<400: reasons.append(f"small_{int(r['width'])}x{int(r['height'])}")
    if r["width"]<300 or r["height"]<300: reasons.append(f"tiny_{int(r['width'])}x{int(r['height'])}")
    for rs in reasons:
        add_anom(r, rs)
p("extreme dim/AR anomalies:", len(anom))

# Brightness pass (near-black/near-white) on subset: all images may be slow, do all but via thumbnail
p("Computing brightness (thumbnail) for near-black/white detection...")
bright_rows=[]
cnt=0
for _,r in ok.iterrows():
    try:
        with Image.open(r["path"]) as im:
            im=im.convert("L")
            im.thumbnail((64,64))
            a=np.asarray(im,dtype=np.float32)
            mean=float(a.mean()); std=float(a.std())
            bright_rows.append((r["path"],mean,std))
            if mean<=20: add_anom(r, f"near_black_mean={mean:.1f}")
            if mean>=235: add_anom(r, f"near_white_mean={mean:.1f}")
            if std<=12: add_anom(r, f"low_contrast_std={std:.1f}")
    except Exception as e:
        add_anom(r, f"brightness_error:{e}")
    cnt+=1
    if cnt%1000==0: p("  brightness progress", cnt)
bdf=pd.DataFrame(bright_rows, columns=["path","bright_mean","bright_std"])
bdf.to_csv(os.path.join(ROOT,"brightness.csv"), index=False, encoding="utf-8")

adf=pd.DataFrame(anom)
if len(adf):
    adf=adf.sort_values(["reason","class","split"])
adf.to_csv(os.path.join(ROOT,"anomalies.csv"), index=False, encoding="utf-8")
p("anomalies.csv written rows:", len(adf))
p("anomaly reason counts:")
if len(adf):
    for rs,c in adf["reason"].apply(lambda x:x.split("_AR")[0].split("=")[0].split("_mean")[0].split("_std")[0]).value_counts().items():
        p("   ",rs,c)

# smallest images and most extreme AR (for the 3 cropped suspects)
p("\n-- 20 smallest by area --")
ok["area"]=ok["width"]*ok["height"]
for _,r in ok.nsmallest(20,"area").iterrows():
    p(f"   {int(r['width'])}x{int(r['height'])} AR={r['aspect_ratio']:.3f} {r['class']}/{r['split']} {r['filename']}")
p("\n-- 15 most extreme (tallest) AR --")
for _,r in ok.nsmallest(15,"aspect_ratio").iterrows():
    p(f"   AR={r['aspect_ratio']:.3f} {int(r['width'])}x{int(r['height'])} {r['class']}/{r['split']} {r['filename']}")
p("\n-- 15 widest AR --")
for _,r in ok.nlargest(15,"aspect_ratio").iterrows():
    p(f"   AR={r['aspect_ratio']:.3f} {int(r['width'])}x{int(r['height'])} {r['class']}/{r['split']} {r['filename']}")

with open(os.path.join(ROOT,"eda_stats.txt"),"w",encoding="utf-8") as f:
    f.write("\n".join(out))
print("\nDONE. stats -> eda_stats.txt")

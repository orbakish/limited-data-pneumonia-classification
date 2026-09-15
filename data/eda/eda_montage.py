# -*- coding: utf-8 -*-
"""READ-ONLY: build montage figures for visual inspection."""
import os, sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS=None
ImageFile.LOAD_TRUNCATED_IMAGES=True

ROOT=os.path.dirname(os.path.abspath(__file__))
FIG=os.path.join(ROOT,"eda_figures"); os.makedirs(FIG,exist_ok=True)
df=pd.read_csv(os.path.join(ROOT,"image_inventory.csv"))
ok=df[df["status"]=="ok"].copy()
ok["area"]=ok["width"].astype(float)*ok["height"].astype(float)

def montage(paths, titles, fname, cols=4, title=""):
    n=len(paths); rows=(n+cols-1)//cols
    plt.figure(figsize=(cols*3, rows*3.2))
    for i,(p_,t_) in enumerate(zip(paths,titles)):
        ax=plt.subplot(rows,cols,i+1)
        try:
            im=Image.open(p_).convert("L")
            ax.imshow(np.asarray(im),cmap="gray")
        except Exception as e:
            ax.text(0.5,0.5,str(e),fontsize=6)
        ax.set_title(t_,fontsize=7)
        ax.axis("off")
    if title: plt.suptitle(title,fontsize=11)
    plt.tight_layout(); plt.savefig(os.path.join(FIG,fname),dpi=110); plt.close()
    print("saved",fname)

# 1. smallest / most cropped (likely the 3 cropped)
sm=ok.nsmallest(12,"area")
montage(list(sm["path"]),[f"{r['filename']}\n{int(r['width'])}x{int(r['height'])} AR{r['aspect_ratio']:.2f}" for _,r in sm.iterrows()],
        "insp_smallest.png",title="12 smallest-area images (crop suspects)")

# 2. widest AR
wide=ok.nlargest(12,"aspect_ratio")
montage(list(wide["path"]),[f"{r['filename']}\n{int(r['width'])}x{int(r['height'])} AR{r['aspect_ratio']:.2f}" for _,r in wide.iterrows()],
        "insp_widest.png",title="12 widest aspect-ratio images")

# 3. random NORMAL sample
rn=ok[ok["class"]=="NORMAL"].sample(12,random_state=1)
montage(list(rn["path"]),[f"{r['split']} {r['filename']}\n{int(r['width'])}x{int(r['height'])}" for _,r in rn.iterrows()],
        "insp_normal_sample.png",title="Random NORMAL sample")

# 4. random PNEUMONIA sample
rp=ok[ok["class"]=="PNEUMONIA"].sample(12,random_state=1)
montage(list(rp["path"]),[f"{r['split']} {r['filename']}\n{int(r['width'])}x{int(r['height'])}" for _,r in rp.iterrows()],
        "insp_pneumonia_sample.png",title="Random PNEUMONIA sample")

# 5. RGB images (all in PNEUMONIA)
rgb=ok[ok["mode"]=="RGB"].sample(min(12,len(ok[ok['mode']=='RGB'])),random_state=1)
montage(list(rgb["path"]),[f"{r['filename']}\n{int(r['width'])}x{int(r['height'])}" for _,r in rgb.iterrows()],
        "insp_rgb.png",title="RGB PNEUMONIA samples")

# 6. duplicate example groups
dups=ok.groupby("hash").filter(lambda g:len(g)>1).sort_values("hash")
dg=dups.groupby("hash").head(2).head(12)
montage(list(dg["path"]),[f"{r['filename']}" for _,r in dg.iterrows()],
        "insp_duplicates.png",title="Exact-duplicate examples (pairs)")

# 7. multi-image PNEUMONIA patient (person1481 series)
series=ok[ok["filename"].str.startswith("person1481_")].sort_values("filename")
if len(series):
    montage(list(series["path"]),[f"{r['filename']}" for _,r in series.iterrows()],
        "insp_person1481.png",title="person1481 multi-image series")
print("DONE")

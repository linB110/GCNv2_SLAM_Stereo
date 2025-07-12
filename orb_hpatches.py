import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

dataset_root = '/home/lab605/lab605/dataset/HPatches'
orb = cv2.ORB_create(nfeatures=3000)
#orb = cv2.ORB_create()

for seq_path in tqdm(sorted(Path(dataset_root).iterdir())):
    if not seq_path.is_dir():
        continue

    for i in range(1, 7):
        img_path = seq_path / f'{i}.ppm'
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Failed to load image: {img_path}")
            continue
        
        kpts = orb.detect(img, None)
        kpts, descs = orb.compute(img, kpts)

        if descs is None or len(kpts) == 0:
            print(f"⚠️ No ORB features in {img_path}")
            continue

        keypoints = np.array([kp.pt for kp in kpts], dtype=np.float32)

        out_dir = seq_path / f"{i}.ppm.orb"
        out_dir.mkdir(exist_ok=True)

        try:
            np.save(out_dir / "keypoints.npy", keypoints)
            np.save(out_dir / "descriptors.npy", descs)
            print(f"✅ Saved to {out_dir}")
        except Exception as e:
            print(f"❌ Failed to save to {out_dir}: {e}")


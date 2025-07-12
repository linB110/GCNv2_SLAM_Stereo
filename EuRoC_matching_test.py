import os
import cv2
import torch
import numpy as np
from glob import glob
from tqdm import tqdm

# === CONFIG ===
LEFT_DIR = "/home/lab605/lab605/dataset/EuRoC/MH_01/mav0/cam0/data"
RIGHT_DIR = "/home/lab605/lab605/dataset/EuRoC/MH_01/mav0/cam1/data"
GCN_PATH = "/home/lab605/lab605/GCNv2_SLAM/GCN2/gcn2_320x240.pt"
OUTPUT_DIR = "./euroc_gcn_matches"

NUM_PAIRS = 100
IMG_WIDTH = 320
IMG_HEIGHT = 240
MAX_KP = 1000
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD MODEL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load(GCN_PATH, map_location=device)
model.eval().to(device)

# === FUNCTIONS ===
def preprocess_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    original_size = img.shape
    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img_tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    return img_resized, img_tensor, original_size

def extract_features(tensor, max_kp=MAX_KP):
    with torch.no_grad():
        keypoints_tensor, desc_tensor = model(tensor)
        if keypoints_tensor.shape[0] == 0:
            return [], None
        if keypoints_tensor.shape[1] == 3:
            scores = keypoints_tensor[:, 2]
            idx = torch.argsort(scores, descending=True)[:max_kp]
            keypoints_tensor = keypoints_tensor[idx]
            desc_tensor = desc_tensor[idx]
        else:
            keypoints_tensor = keypoints_tensor[:max_kp]
            desc_tensor = desc_tensor[:max_kp]
        pts = keypoints_tensor[:, :2].cpu().numpy()
        desc = desc_tensor.cpu().numpy()
        kps = [cv2.KeyPoint(pt[0], pt[1], 1) for pt in pts]
        return kps, desc.astype(np.float32)

def match_descriptors(desc1, desc2):
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    return matcher.match(desc1, desc2)

def draw_and_save(img1, kp1, img2, kp2, matches, name):
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:100], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_match.png"), match_img)

# === MAIN LOOP ===
left_imgs = sorted(glob(os.path.join(LEFT_DIR, "*.png")))[:NUM_PAIRS]
right_imgs = sorted(glob(os.path.join(RIGHT_DIR, "*.png")))[:NUM_PAIRS]

for l_path, r_path in tqdm(zip(left_imgs, right_imgs), total=NUM_PAIRS):
    name = os.path.splitext(os.path.basename(l_path))[0]
    try:
        imgL, tensorL, _ = preprocess_img(l_path)
        imgR, tensorR, _ = preprocess_img(r_path)

        kpL, descL = extract_features(tensorL)
        kpR, descR = extract_features(tensorR)

        if not kpL or not kpR:
            print(f"[WARN] No keypoints: {name}")
            continue

        matches = match_descriptors(descL, descR)
        draw_and_save(imgL, kpL, imgR, kpR, matches, name)

        print(f"[OK] Matched {len(matches)} features in {name}")
    except Exception as e:
        print(f"[ERROR] Failed at {name}: {e}")


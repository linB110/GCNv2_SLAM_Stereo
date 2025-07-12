import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

MAX_KEYPOINT = 1000

# -------- functions --------
def load_and_preprocess(img_path):
    img_uint8 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
    img_resized = cv2.resize(img_uint8, (320, 240))
    img = img_resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
    return img_resized, tensor  

def extract_features(model, img_tensor, max_keypoints=MAX_KEYPOINT):
    with torch.no_grad():
        keypoints_tensor, desc = model(img_tensor)

        if keypoints_tensor.shape[0] == 0:
            return [], [], None

        keypoints_tensor = keypoints_tensor[:max_keypoints]
        desc = desc[:max_keypoints]

        pts = keypoints_tensor[:, :2].cpu().numpy()
        desc = desc.cpu().numpy()

        keypoints = [cv2.KeyPoint(pt[0], pt[1], 1) for pt in pts]
        return keypoints, desc, None

def match_descriptors(desc1, desc2):
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.match(desc1.astype(np.float32), desc2.astype(np.float32))
    return sorted(matches, key=lambda x: x.distance)

def draw_and_save_matches(img1, kp1, img2, kp2, matches, out_path, top_k=MAX_KEYPOINT):
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:top_k], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(out_path, match_img)

def compute_matching_accuracy(matches, kp1, kp2, H, threshold=3):
    correct = 0
    total = len(matches)
    for m in matches:
        pt1 = np.array([*kp1[m.queryIdx].pt, 1.0])
        projected = H @ pt1
        projected /= projected[2]
        pt2 = np.array(kp2[m.trainIdx].pt)
        error = np.linalg.norm(projected[:2] - pt2)
        if error < threshold:
            correct += 1
    return correct / total if total > 0 else 0

def compute_mean_accuracy(values):
    return sum(values) / len(values) if values else 0

def plot_histogram(illum_results, view_results, save_path):
    illum_max = max(illum_results) if illum_results else 0
    illum_min = min(illum_results) if illum_results else 0
    illum_mean = compute_mean_accuracy(illum_results)

    view_max = max(view_results) if view_results else 0
    view_min = min(view_results) if view_results else 0
    view_mean = compute_mean_accuracy(view_results)

    labels = ['Max', 'Min', 'Mean']
    illum_values = [illum_max, illum_min, illum_mean]
    view_values = [view_max, view_min, view_mean]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, illum_values, width, label='Illumination', color='orange')
    bars2 = ax.bar(x + width/2, view_values, width, label='Viewpoint', color='blue')

    ax.set_ylabel('Matching Accuracy')
    ax.set_title('HPatches Matching Accuracy Statistics (GCNv2)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis='y')

    def annotate_bars(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1%}',  # e.g. 28.9%
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    annotate_bars(bars1)
    annotate_bars(bars2)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# -------- main --------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.jit.load('/home/lab605/lab605/GCNv2_SLAM/GCN2/gcn2_320x240.pt', map_location=device)
    model.to(device)
    model.eval()

    hpatches_root = '/home/lab605/lab605/dataset/HPatches'
    output_root = '/home/lab605/lab605/GCNv2_SLAM/GCN_matching'
    os.makedirs(output_root, exist_ok=True)

    results = []
    illumination_results = []
    viewpoint_results = []

    seqs = sorted(glob(os.path.join(hpatches_root, '*')))
    for seq in seqs:
        name = os.path.basename(seq)
        img1_path = os.path.join(seq, '1.ppm')
        img2_path = os.path.join(seq, '6.ppm')
        H_path = os.path.join(seq, 'H_1_6')

        if not all(os.path.exists(p) for p in [img1_path, img2_path, H_path]):
            print(f"❌ lack of {name} file, omitting")
            continue

        img1, tensor1 = load_and_preprocess(img1_path)
        img2, tensor2 = load_and_preprocess(img2_path)

        try:
            kp1, desc1, _ = extract_features(model, tensor1)
            kp2, desc2, _ = extract_features(model, tensor2)
        except Exception as e:
            print(f"❌ extract feature failed {name}: {e}")
            continue

        if len(kp1) == 0 or len(kp2) == 0:
            print(f"⚠️ can't detect keypoint {name}, skipping")
            continue

        matches = match_descriptors(desc1, desc2)
        H = np.loadtxt(H_path)
        acc = compute_matching_accuracy(matches, kp1, kp2, H)

        if name.startswith('i'):
            illumination_results.append(acc)
        elif name.startswith('v'):
            viewpoint_results.append(acc)

        print(f"{name}: Matching Accuracy = {acc:.4f}")
        results.append((name, acc))

        draw_and_save_matches(img1, kp1, img2, kp2, matches,
                              os.path.join(output_root, f"{name}_matches.png"))

    with open(os.path.join(output_root, "matching_results.txt"), 'w') as f:
        for name, acc in results:
            f.write(f"{name}: {acc:.4f}\n")

    illum_avg = compute_mean_accuracy(illumination_results)
    view_avg = compute_mean_accuracy(viewpoint_results)
    print("\n📊 Average Matching Accuracy:")
    print(f"Illumination: {illum_avg:.4f}")
    print(f"Viewpoint:    {view_avg:.4f}")

    plot_histogram(illumination_results, viewpoint_results,
                   os.path.join(output_root, 'matching_accuracy_histogram.png'))

    print("✅ task finished！")


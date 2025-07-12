import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

MAX_KEYPOINT = 1000
OUTPUT_METHOD = 'gcnv2_aug'

def load_and_preprocess(img_path):
    img_uint8 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_uint8 is None:
        raise ValueError(f"Cannot load image {img_path}")

    original_h, original_w = img_uint8.shape
    original_size = (original_w, original_h)

    img_resized = cv2.resize(img_uint8, (320, 240))
    img = img_resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
    return img_resized, tensor, original_size

def extract_features(model, img_tensor, max_keypoints=MAX_KEYPOINT):
    with torch.no_grad():
        keypoints_tensor, desc = model(img_tensor)

        if keypoints_tensor.shape[0] == 0:
            return [], [], None

        if keypoints_tensor.shape[1] == 3:
            scores = keypoints_tensor[:, 2]
            idx = torch.argsort(scores, descending=True)[:max_keypoints]
            keypoints_tensor = keypoints_tensor[idx]
            desc = desc[idx]
        else:
            keypoints_tensor = keypoints_tensor[:max_keypoints]
            desc = desc[:max_keypoints]

        pts = keypoints_tensor[:, :2].cpu().numpy()
        desc = desc.cpu().numpy().astype(np.float32)
        desc /= np.linalg.norm(desc, axis=1, keepdims=True) + 1e-8

        keypoints = [cv2.KeyPoint(pt[0], pt[1], 1) for pt in pts]

        if desc.shape[0] == 0:
            return [], [], None

        return keypoints, desc, pts

def match_descriptors(desc1, desc2):
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.match(desc1.astype(np.float32), desc2.astype(np.float32))
    return sorted(matches, key=lambda x: x.distance)

def draw_and_save_matches(img1, kp1, img2, kp2, matches, out_path, top_k=MAX_KEYPOINT):
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:top_k], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(out_path, match_img)

def save_features(seq_name, img_idx, kpts, desc, original_size, output_dir):
    original_w, original_h = original_size
    sx = original_w / 320
    sy = original_h / 240

    kpts_rescaled = np.array([[pt[0] * sx, pt[1] * sy] for pt in kpts])  # (x, y)
    save_dir = os.path.join(output_dir, seq_name, f"{img_idx}.ppm.{OUTPUT_METHOD}")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "keypoints.npy"), kpts_rescaled)
    np.save(os.path.join(save_dir, "descriptors.npy"), desc)


def compute_matching_accuracy(matches, kp1, kp2, H, threshold=3):
    correct = 0
    for m in matches:
        pt1 = np.array([*kp1[m.queryIdx].pt, 1.0])
        projected = H @ pt1
        projected /= projected[2]
        pt2 = np.array(kp2[m.trainIdx].pt)
        error = np.linalg.norm(projected[:2] - pt2)
        if error < threshold:
            correct += 1
    return correct / len(matches) if matches else 0

def compute_mean_accuracy(values):
    return sum(values) / len(values) if values else 0

def plot_histogram(illum_results, view_results, save_path):
    illum_max = max(illum_results) if illum_results else 0
    illum_min = min(illum_results) if illum_results else 0
    illum_mean = compute_mean_accuracy(illum_results)

    view_max = max(view_results) if view_results else 0
    view_min = min(view_results) if view_results else 0
    view_mean = compute_mean_accuracy(view_results)

    labels = ['Illum Max', 'Illum Min', 'Illum Mean', 'View Max', 'View Min', 'View Mean']
    values = [illum_max, illum_min, illum_mean, view_max, view_min, view_mean]
    colors = ['orange'] * 3 + ['blue'] * 3

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=colors)
    plt.ylim(0, 1)
    plt.ylabel('Matching Accuracy')
    plt.title('HPatches Matching Accuracy Statistics (GCNv2)')
    plt.grid(axis='y')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                 f'{height*100:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device =", device)

    model = torch.jit.load('/home/lab605/lab605/GCNv2_SLAM/GCN2/gcn2_aug.pt', map_location=device)
    model.to(device)
    model.eval()

    hpatches_root = '/home/lab605/lab605/dataset/HPatches'
    output_root = '/home/lab605/lab605/GCNv2_SLAM/GCN_matching'
    os.makedirs(output_root, exist_ok=True)

    illumination_results = []
    viewpoint_results = []
    results = []

    seqs = sorted(glob(os.path.join(hpatches_root, '*')))
    for seq in seqs:
        name = os.path.basename(seq)
        for idx in range(1, 7):
            img_path = os.path.join(seq, f"{idx}.ppm")
            if not os.path.exists(img_path):
                continue
            img, tensor, original_size = load_and_preprocess(img_path)
            try:
                kp, desc, pts = extract_features(model, tensor)
                if len(kp) == 0:
                    continue
                save_features(name, idx, pts, desc, original_size, hpatches_root)
            except Exception as e:
                print(f"⚠️ {name}-{idx} feature extract failed: {e}")
                continue

        img1_path = os.path.join(seq, '1.ppm')
        img2_path = os.path.join(seq, '6.ppm')
        H_path = os.path.join(seq, 'H_1_6')
        if not all(os.path.exists(p) for p in [img1_path, img2_path, H_path]):
            print(f"❌ Missing file in {name}, skipping")
            continue

        img1, tensor1, size1 = load_and_preprocess(img1_path)
        img2, tensor2, size2 = load_and_preprocess(img2_path)

        try:
            kp1, desc1, _ = extract_features(model, tensor1)
            kp2, desc2, _ = extract_features(model, tensor2)
        except Exception as e:
            print(f"❌ Extract feature failed {name}: {e}")
            continue

        if len(kp1) == 0 or len(kp2) == 0:
            print(f"⚠️ No keypoints in {name}, skipping")
            continue

        matches = match_descriptors(desc1, desc2)

        H = np.loadtxt(H_path)
        sx = 320 / size1[0]
        sy = 240 / size1[1]
        S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
        H = S @ H @ np.linalg.inv(S)

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

    print("\n📊 Average Matching Accuracy:")
    print(f"Illumination: {compute_mean_accuracy(illumination_results):.4f}")
    print(f"Viewpoint:    {compute_mean_accuracy(viewpoint_results):.4f}")

    plot_histogram(illumination_results, viewpoint_results,
                   os.path.join(output_root, 'matching_accuracy_histogram.png'))

    print("✅ GCNv2 evaluation finished!")

# 🌀 GCNv2\_SLAM-Stereo

A modified version of [GCNv2\_SLAM](https://github.com/jiexiong2016/GCNv2_SLAM) and [ORB\_SLAM2](https://github.com/raulmur/ORB_SLAM2) with **stereo camera** support and enhanced setup instructions.
This project enables real-time stereo visual SLAM using the GCNv2 keypoint extractor.

---

## 🚀 Features

* ✅ Added **stereo camera support**
* ✅ Enhanced build instructions and Python environment setup
* ✅ Support for [TUM](https://vision.in.tum.de/data/datasets/rgbd-dataset) / [EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)
* ✅ Integrated GCNv2 model accuracy visualizer
* ✅ Compatible with `evo_ape` evaluation

---

## 🛠 Environment & Dependencies

Tested on:

* **OS**: Ubuntu 18.04 LTS
* **GPU**: NVIDIA GeForce RTX 2060
* **CUDA**: 10.2
* **cuDNN**: Compatible with CUDA 10.2
* **PyTorch**: 1.9.1 (built manually)
* **CPU**: Intel Xeon E3-1230 V2

### 📦 Required Downloads

* **libtorch (PyTorch C++ 1.9.1)**

```bash
wget https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.9.1+cu102.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.9.1+cu102.zip
```

* **ORB\_SLAM2**
  Follow setup guide: [https://github.com/raulmur/ORB\_SLAM2](https://github.com/raulmur/ORB_SLAM2)

---

## 🔧 Build GCNv2\_SLAM

Make sure you're using the modified `CMakeLists.txt` provided in this repo.

Then build the project:

```bash
./build.sh
```

---

## 🥪 Test GCN Feature Extractor

Create a Python environment:

```bash
conda create -n gcnv2_env python=3.8 -y
conda activate gcnv2_env
conda install pytorch==1.10.2 torchvision==0.11.3 cudatoolkit=10.2 -c pytorch
pip install opencv-python matplotlib
```

Run visualization:

```bash
python show_accuracy.py
```

### Output

* A folder named `GCN_matching` will be created.
* This folder contains keypoint matching visualizations.

---

## 🎮 Run GCNv2\_SLAM on Dataset

### Model

Use the provided model file:

```
model/gcn2_320x240.pt
```

### Create Association File

```bash
python associate.py
```

### Run SLAM (TUM)

```bash
cd ~/GCN2

GCN_PATH=/home/lab605/lab605/GCNv2_SLAM/GCN2/gcn2_320x240.pt ./rgbd_gcn \
    /home/lab605/lab605/GCNv2_SLAM/Vocabulary/GCNvoc.bin \
    /home/lab605/lab605/GCNv2_SLAM/GCN2/TUM3.yaml \
    /home/lab605/lab605/dataset/TUM/rgbd_dataset_freiburg1_xyz \
    /home/lab605/lab605/dataset/TUM/rgbd_dataset_freiburg1_xyz/association.txt
```

### Run SLAM (EuRoC)

```bash
cd ~/GCN2

 GCN_PATH=/home/lab605/lab605/GCNv2_SLAM/GCN2/gcn2_320x240.pt
    /home/lab605/lab605/GCNv2_SLAM/GCN2/stereo_gcn /home/lab605/lab605/GCNv2_SLAM/Vocabulary/GCNvoc.bin
    /home/lab605/lab605/ORB_SLAM2/Examples/Stereo/EuRoC.yaml
    /home/lab605/lab605/dataset/EuRoC/MH_01/mav0/cam0/data
    /home/lab605/lab605/dataset/EuRoC/MH_01/mav0/cam1/data
    /home/lab605/lab605/ORB_SLAM2/Examples/Stereo/EuRoC_TimeStamps/MH01.txt
```

Make sure to update paths according to your system.

---

## 📊 Evaluation with `evo`

Evaluate Absolute Pose Error using [`evo`](https://github.com/MichaelGrupp/evo):

```bash
evo_ape tum \
    /home/lab605/lab605/dataset/TUM/rgbd_dataset_freiburg1_xyz/groundtruth.txt \
    /home/lab605/lab605/GCNv2_SLAM/GCN2/KeyFrameTrajectory.txt \
    --align --plot
```

---

## 📁 Suggested Folder Structure

```
GCNv2_SLAM-Stereo/
├── GCN2/
│   ├── gcn2_320x240.pt
│   ├── TUM3.yaml
│   ├── KeyFrameTrajectory.txt
├── Vocabulary/
│   └── GCNvoc.bin
├── dataset/
│   └── TUM/...
├── build.sh
├── associate.py
├── show_accuracy.py
├── CMakeLists.txt
└── ...
```

---

---
## 📊 Evaluation 
✅ : fully track along wohle sequence

❌ : lost track during thje sequence

| Dataset / Sequence                                       | ORB (nlevel=8) | ORB (nlevel=1) | 320x240.pt  | 640x480.pt  | aug.pt  | tiny  |
|-----------------------------------------------------------|------|----------------|-------------|-------------|---------|-------|
| **TUM RGB-D**                                             |      |                |             |             |         |       |
| rgbd_dataset_freiburg1_desk                               | 0.018781 ✅ |  0.014945 ✅           | 0.036593 ❌  | 0.132776 ✅  | 0.020748 ✅ | 0.222797 ✅ |
| rgbd_dataset_freiburg1_xyz                                | 0.012081 ✅ |  0.009779 ✅            | 0.080495 ✅  | 0.014587 ✅  | 0.088993 ✅ | 0.084459 ✅ |
| rgbd_dataset_freiburg2_pioneer_360                        | 0.065314 ✅ | 0.028517 ✅          | 0.307028 ❌  | 0.585210 ❌  | 0.323715 ✅ | 0.079713 ❌ |
| rgbd_dataset_freiburg3_nostructure_notexture_near_withloop | ❌         |  ❌           | ❌          | 0.004605 ❌  | ❌       | ❌     |
| **EuRoC Stereo**                                          |      |      |             |             |         |       |
| MH_01                                                     | 0.037540 ✅ |  -            | 0.018083 ❌  | 0.175981 ✅  | 0.014021 ❌ | 0.038764 ❌ |
| MH_05                                                     | 0.047538 ✅ |  -            | 0.544578 ❌  | 0.892108 ❌  | 0.558102 ❌ | 0.052919 ❌ |


---

## 🙏 Acknowledgements

This project is based on:

* [ORB\_SLAM2](https://github.com/raulmur/ORB_SLAM2)
* [GCNv2\_SLAM](https://github.com/jiexiong2016/GCNv2_SLAM)

Stereo support and additional tools were added in this fork.

---

## 📌 Notes

* All dataset and model paths must be valid.
* It's strongly recommended to use isolated Conda environments.
* Pull requests are welcome if you wish to contribute!


Run SLAM
```bash

FULL_RESOLUTION=1
GCN_PATH=/home/lab605/lab605/GCNv2_SLAM/GCN2/gcn2_640x480.pt ./rgbd_gcn
/home/lab605/lab605/GCNv2_SLAM/Vocabulary/GCNvoc.bin
/home/lab605/lab605/ORB_SLAM2/Examples/Stereo/EuRoC.yaml
/home/lab605/lab605/dataset/TUM/rgbd_dataset_freiburg2_pioneer_360
/home/lab605/lab605/dataset/TUM/rgbd_dataset_freiburg2_pioneer_360/association.txt


```

Run Evaluation

```bash

 evo_ape tum /home/lab605/lab605/dataset/TUM/groundtruth_tum.txt
/home/lab605/lab605/ORB_SLAM2/CameraTrajectory.txt --align --correct_scale --plot --save_plot ate_desk.pdf

```


Run SLAM
```bash

FULL_RESOLUTION=1 
GCN_PATH=/home/lab605/lab605/GCNv2_SLAM/GCN2/gcn2_640x480.pt
 /home/lab605/lab605/GCNv2_SLAM/GCN2/stereo_gcn /home/lab605/lab605/GCNv2_SLAM/Vocabulary/GCNvoc.bin
/home/lab605/lab605/ORB_SLAM2/Examples/Stereo/EuRoC.yaml
 /home/lab605/lab605/dataset/EuRoC/MH_01/mav0/cam0/data   /home/lab605/lab605/dataset/EuRoC/MH_01/mav0/cam1/data
/home/lab605/lab605/ORB_SLAM2/Examples/Stereo/EuRoC_TimeStamps/MH01.txt

```

Run evaluation
```bash

evo_ape tum /home/lab605/lab605/dataset/EuRoC/MH_01/groundtruth_tum.txt
/home/lab605/lab605/GCNv2_SLAM/GCN2/CameraTrajectory.txt --align --correct_scale --plot --save_plot ate_MH01.pdf


```

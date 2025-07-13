Run SLAM
```bash

/home/lab605/lab605/ORB_SLAM2/Examples/RGB-D/rgbd_tum
/home/lab605/lab605/ORB_SLAM2/Vocabulary/ORBvoc.txt
/home/lab605/lab605/ORB_SLAM2/Examples/RGB-D/TUM1.yaml
 /home/lab605/lab605/dataset/TUM/rgbd_dataset_freiburg1_desk
/home/lab605/lab605/dataset/TUM/rgbd_dataset_freiburg1_desk/association.txt

```

Run Evaluation

```bash

 evo_ape tum /home/lab605/lab605/dataset/TUM/groundtruth_tum.txt
/home/lab605/lab605/ORB_SLAM2/CameraTrajectory.txt --align --correct_scale --plot --save_plot ate_desk.pdf

```


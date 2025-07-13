Run SLAM
```bash

/home/lab605/lab605/ORB_SLAM2/Examples/Stereo/stereo_euroc
 /home/lab605/lab605/ORB_SLAM2/Vocabulary/ORBvoc.txt
 /home/lab605/lab605/ORB_SLAM2/Examples/Stereo/EuRoC.yaml
 /home/lab605/lab605/dataset/EuRoC/MH_01/mav0/cam0/data /home/lab605/lab605/dataset/EuRoC/MH_01/mav0/cam1/data
/home/lab605/lab605/ORB_SLAM2/Examples/Stereo/EuRoC_TimeStamps/MH01.txt


```
Run Evaluation
```bash

 evo_ape tum /home/lab605/lab605/dataset/EuRoC/MH_01/groundtruth_tum.txt
/home/lab605/lab605/ORB_SLAM2/CameraTrajectory.txt --align --correct_scale --plot --save_plot ate_MH01.pdf


```

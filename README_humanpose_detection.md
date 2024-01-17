## Summary for running human pose detection along with AprilTag Detection

# For Recording HandPose data
```bash
conda activate openmmlab

cd ~/repos/mmpose

python  demo/topdown_demo_with_mmdet_multicam.py demo/mmdetection_cfg/rtmdet_nano_320-8xb32_hand.py \
        https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_nano_8xb32-300e_hand-267f9c8f.pth  \
        configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py  \
        https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth   \
        --output_data_folder {folder_name - /scratch/cchawla/RigidBodyHumanData/test/}  \
        --kpt-thr 0.1 \
        --task_name {task_name} \
        --primary_camera 0  

# ctrl+c to stop and save the recording
```


# For RViz Vizualization

```bash
conda activate shape2

cd ~/repos/ros-shape2shape/catkin_ws/

roslaunch xarm_description human_demo_display.launch folder_name:={numpy_file_location}  from_dataset:={True/False} visualize_cam_num:={0 or 1}
```

# ###########################################################################
# The approach that we used ----
# For first recording images and then processing them offline

```bash
conda activate openmmlab

cd ~/repos/mmpose

python  demo/topdown_demo_with_mmdet_multicam_images.py demo/mmdetection_cfg/rtmdet_nano_320-8xb32_hand.py \
        https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_nano_8xb32-300e_hand-267f9c8f.pth  \
        configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py  \
        https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth
        --input /scratch/cchawla/RigidBodyHumanData/Images/BoxOpening/

# After recording all images, separate folders containing numpy files can be processed - 

python  demo/extract_handpose_from_images.py demo/mmdetection_cfg/rtmdet_nano_320-8xb32_hand.py \
        https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_nano_8xb32-300e_hand-267f9c8f.pth  \
        configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py  \
        https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth
        --input {folder_name: /scratch/cchawla/RigidBodyHumanData/Images/{task_name} }


# Data in /scratch/cchawla/RigidBodyHumanData/Images/Processed_Demos/BoxOpening is same as the data in /scratch/cchawla/RigidBodyHumanData/Images/BoxOpening/processed_demos. Applies for other tasks as well

# This was done to have more organized data for the dataloader.
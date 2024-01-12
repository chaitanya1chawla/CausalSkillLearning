import numpy as np
import cv2
import glob
import os

folder_name = '/scratch/cchawla/RigidBodyHumanData/Images/BoxOpening/unprocessed_demos'

for idx, file in enumerate(sorted(glob.glob(os.path.join(folder_name, '*.npy')))):

    x=np.load(file, allow_pickle=True)
    print('File name = ', file)
    print('primary_camera= ', x[1]['primary_camera'])
    print('primary_keypoint_camera= ', x[1]['primary_keypoint_camera'])
    
    # if file == '/scratch/cchawla/RigidBodyHumanData/Images/BoxOpening/demo_08122023_232358.npy':
        # continue
    
    # Extract images  here -- 
    # for cam in ['cam0', 'cam1']:
    #     ctr=0

    #     for img in x[1]['images'][cam]:
    #         image_folder = os.path.join(folder_name, 'traj{}_images'.format(idx+1), cam)
    #         image_file = os.path.join(image_folder, 'pic{}.png'.format(ctr))
    #         cv2.imwrite(image_file, img)
    #         ctr += 1

    x[1]['primary_keypoint_camera'] = 1
    print('new primary_keypoint_camera= ', x[1]['primary_keypoint_camera'])
    
    # file = file[:-4] + '_new' + file[-4:]
    file = file[:-42] + 'new_' + file[-42:]
    print('new File name = ', file)

    np.save(file, x)

    del x
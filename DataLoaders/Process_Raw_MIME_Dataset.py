import subprocess, os, glob

task_hash = \
['AADypxdDq9CbeD4VRKulTbG1a\?dl\=0',
 'AAD_0YyCm17oradIgtoTEidja\?dl\=0',
 'AACZJ25JmluVCeDU7ZrnMRqNa\?dl\=0',
 'AAD2GKHwT7OtA4rtYxCnViNKa\?dl\=0',
 'AAC4qTLuAyYq5EXccBUuuzt6a\?dl\=0',
 'AAA2BKWxRpXgyy8R96bRRU1ua\?dl\=0',
 'AADU0alwAPCxRY0qfNS_O0Cza\?dl\=0',
 'AAC4GB-uPSgtrPsJpAfNYjJta\?dl\=0',
#  'AADOg_NUFI_oU6lR3E0mZhkQa\?dl\=0',
 'AAAguCBiKiEh8Aem2HtjXDcKa\?dl\=0',
 'AAB4zokOUUA2x2QIEGrZUawNa\?dl\=0',
 'AACb912BIksbG1PWHO0T8GMYa\?dl\=0',
 'AABBHku0_4EGyXH3__vAyijTa\?dl\=0',
 'AACDSbpiCRUIwfNWjI4lyydya\?dl\=0',
 'AACG-uJ6OYLwvwC0RngIMI5ja\?dl\=0',
 'AADln-EXkyVCCRU_XM75qlfSa\?dl\=0',
 'AACgj6_NIjxJC1dKgK35TsJSa\?dl\=0',
 'AACBW2NQ1WAcj227GWm1wYfoa\?dl\=0',
 'AAAlwwPFXWd-MggI62uEwCwna\?dl\=0',
 'AAAAjU_yA79odQMGgYtVWJA6a\?dl\=0',
 'AAB71YePoQfovylmMzZILWSxa\?dl\=0']

basepath = '/data1/tanmayshankar/MIME_FullDataset'
joint_basepath = '/data1/tanmayshankar/MIME_JointDataset'

# # First, unzip everything
# for i in range(20):    
    
#     print("Processing Task ",i)
    
#     # # First make target directory
#     # os.mkdir(os.path.join(joint_basepath,"Task_{0}".format(i)))
    
#     # Next move to full dataset dir
#     os.chdir(os.path.join(basepath,"Task_{0}".format(i)))
#     # os.chdir(os.path.join(joint_basepath,"Task_{0}".format(i)))
    
#     # Unzip, but ignore videos
#     # command = '7z x {0}'.format(task_hash[i])
#     # command = '7z x {0} -i*.mp4'.format(task_hash[i])
#     command = '7z x {0} -xr!*.mp4'.format(task_hash[i])
#     subprocess.run([command],shell=True)

# Now that things are unzipped.  
os.chdir(basepath)

# Move text files to JointDataset directory
command = 'cp --parents */*/*.txt {0}'.format(joint_basepath)
subprocess.run([command],shell=True)


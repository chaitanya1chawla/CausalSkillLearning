from PolicyNetworks import *

x = np.load("MIMEDataArray_EEAugmented.npy",allow_pickle=True)

# Train network to predict joint states given the EE pose.. 

# 
IK_predictor = ContinuousMLP()
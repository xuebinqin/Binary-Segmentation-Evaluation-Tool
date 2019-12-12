# MAE, Precision, Recall, F-measure, IoU, Precision-Recall curves

import numpy as np

def mask_normalize(mask):
# input 'mask': HxW
# output: HxW [0,255]
    return mask/(np.amax(mask)+1e-8)

def compute_mae(mask1,mask2):
# input 'mask1': HxW or HxWxn
#       'mask2': HxW or HxWxn
# output: a value MAE, Mean Absolute Error
    if(len(mask1.shape)<2 or len(mask2.shape)<2):
        print("ERROR: Mask1 or mask2 is not matrix!")
        exit()
    if(len(mask1.shape)>2):
        mask1 = mask1[:,:,0]
    if(len(mask2.shape)>2):
        mask2 = mask2[:,:,0]
    if(mask1.shape!=mask2.shape):
        print("ERROR: The shapes of mask1 and mask2 are different!")
        exit()
    mask1 = mask_normalize(mask1)
    mask2 = mask_normalize(mask2)
    sumError = np.sum(np.absolute((mask1.astype(float) - mask2.astype(float))))
    maeError = sumError/(float(h)*float(w)*255.0+1e-8)

    return maeError

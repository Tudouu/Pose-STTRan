import cv2
import numpy as np
import torch
import imageio
import copy
from openpose.src.body import Body
from openpose.src.estimate_pose import estimate_bodypose
from GCN import extract_feature_old
#from GCN.extract_feature import extract






root_path="/media/wow/disk2/AG/dataset/frames/"
pi="0ACZ8.mp4/000386.png"
real=root_path+pi
image=cv2.imread(real)
c=list()
a=np.array([[[239.0, 117.0,0.56], [216.0, 127.0,0.86], [192.0, 128.0,0.12], [178.0, 186.0,0.84], [228.0, 187.0,0.96]]])
#print(len(a[0]))

#data=extract(a,480,360)
#print(data)
"""body_estimate=Body("openpose/model/body_pose_model.pth")
picture_path='/media/wow/disk2/AG/dataset/frames/V95RI.mp4/000466.png'
picture=cv2.imread(picture_path)
candidate,subset=body_estimate(picture)
posed_value,pose_index,posed_value_not= estimate_bodypose(candidate, subset)
posed_value=(np.array(posed_value))*2.2

.pose_feature=extract_feature.extract(posed_value,1066,800)"""
#score_order=(-a[:,:,2].sum(axis=1)).argsort(axis=0)
k=np.random.randint(0,10,size=[37,5])
index=np.random.randint(0,36,size=[37])
print(index)
print(k)
print('-----------------------------------')
j=6
print(k[(index==j)+(index==j+1)])
#23 31 36
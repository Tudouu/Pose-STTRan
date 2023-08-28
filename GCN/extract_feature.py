import torch
import numpy as np
import cv2
from openpose.src.body import Body
from openpose.src.estimate_pose import estimate_bodypose
import torchlight
from GCN.model import Model
import collections
from time import *
from light_openpose.models.with_mobilenet import PoseEstimationWithMobileNet
from light_openpose.modules.load_state import load_state
from light_openpose import light_op


def extract(pose,fre_ske_idx):#pose=[rel_num,18,3]
    #rel_num,pose_point,xyz=pose.shape
    #pose=pose.permute(2,0,1)
    #pose=pose.view(xyz,rel_num,pose_point,1)#(3,rel_num,18,1)

    #xx=[]
    #yy=[]
    #ss=[]
    #rel_num,pose_num,l=pose.shape
    #for i in range(rel_num):
    #    l0=pose[i][:,0]
    #    l1=pose[i][:,1]
    #    l2=pose[i][:,2]
    #    xx.append(l0.reshape(18,1))
    #    yy.append(l1.reshape(18,1))
    #    ss.append(l2.reshape(18,1))
    #pose_after.append(xx)
    #pose_after.append(yy)
    #pose_after.append(ss)
    #print(np.array(pose_after).shape)
    #io=torchlight.IO('/media/wow/disk2/STT2/STTran-main/GCN/tmp',True,True,)
    #first_para = 'net.gcn.Model'
    #second_para = {'in_channels': 3, 'num_class': 400, 'edge_importance_weighting': True,
    #               'graph_args': {'layout': 'openpose', 'strategy': 'spatial'}}
    #model=io.load_model(first_para,second_para)
    #model=model.to('cuda:0')
    model=Model(3,{'layout': 'openpose', 'strategy': 'spatial'},True)
    #model=model.to('cuda:0')

    output,feature=model(pose)
    output=output[0]
    feature=feature[0]

    intensity = (feature * feature).sum(dim=0) ** 0.5
    #voting_label = output.sum(dim=3).sum(dim=2).sum(dim=1)
    #print(output.shape)
    #print(feature.shape)
    #print('*',(feature * feature).shape)


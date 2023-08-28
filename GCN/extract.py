import torch
import numpy as np
import cv2
import torchlight
from GCN.gcn import Model
from time import *
from light_openpose.models.with_mobilenet import PoseEstimationWithMobileNet
from light_openpose.modules.load_state import load_state
from light_openpose import light_op


root_path='/media/wow/disk2/AG/dataset/frames/'
net = PoseEstimationWithMobileNet()
checkpoint = torch.load('/media/wow/disk2/STT2/STTran-main/light_openpose/checkpoint_iter_370000.pth', map_location='cpu')
load_state(net, checkpoint)


def extract(openpose_index,openpose_index_len,openposed_value_scale,openpose_rel_man_img_idx):
    fre_ske_idx = []
    a=np.array(openpose_rel_man_img_idx)
    i = 0
    while i < len(a):
        if i == len(a) - 1:
            fre_ske_idx.append(1)
            break
        num = 1
        while a[i] == a[i + 1]:
            num = num + 1
            i = i + 1
            if i == len(a) - 1:
                break
        fre_ske_idx.append(num)
        i = i + 1
    fre_ske_idx=np.array(fre_ske_idx)



    a1=time()
    frame_index = 0

    print(openpose_index)
    for i in range(openpose_index_len):
        image = cv2.imread(root_path + openpose_index[i])
        image=cv2.resize(image,None,None,fx=float(openposed_value_scale),fy=float(openposed_value_scale),interpolation=cv2.INTER_LINEAR)
        w,h,_=image.shape

        multi_pose=light_op.estimate_pose(net,image)
        multi_pose = torch.from_numpy(multi_pose)
        multi_pose = multi_pose.unsqueeze(0)

        multi_pose[:, :, 0] = multi_pose[:, :, 0] / w  # 横坐标
        multi_pose[:, :, 1] = multi_pose[:, :, 1] / h  # 纵坐标
        multi_pose[:, :, 0:2] = multi_pose[:, :, 0:2] - 0.5
        multi_pose[:, :, 0][multi_pose[:, :, 2] == 0] = 0
        multi_pose[:, :, 1][multi_pose[:, :, 2] == 0] = 0

        multi_pose = multi_pose.numpy()
        frame_index += 1


    data_numpy = pose_tracker.get_skeleton_sequence()
    data = torch.from_numpy(data_numpy)
    data = data.unsqueeze(0)
    #print(data.shape)#[1, 3, picture_num , 18, 1]
    data = data.float().to("cuda:0").detach()

    #io=torchlight.IO('/media/wow/disk2/STT2/STTran-main/GCN/tmp',True,True,)
    #first_para = 'net.gcn.Model'
    #second_para = {'in_channels': 3, 'num_class': 400, 'edge_importance_weighting': True,
    #               'graph_args': {'layout': 'openpose', 'strategy': 'spatial'}}
    #model=io.load_model(first_para,second_para)
    #model=model.to('cuda:0')
    model=Model(3,400,{'layout': 'openpose', 'strategy': 'spatial'},True)
    model=model.to('cuda:0')

    output,feature=model(data)
    output=output[0]
    feature=feature[0]

    intensity = (feature * feature).sum(dim=0) ** 0.5
    #voting_label = output.sum(dim=3).sum(dim=2).sum(dim=1)
    #print(output.shape)
    #print(feature.shape)
    #print('*',(feature * feature).shape)
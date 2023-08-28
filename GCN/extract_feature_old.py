import torch
import numpy as np
from GCN.st_gcn_old import Model
import sys
import cv2
import traceback
from collections import OrderedDict
from openpose.src.body import Body
from openpose.src.estimate_pose import estimate_bodypose


root_path='/media/wow/disk2/AG/dataset/frames/'
body_estimate=Body("openpose/model/body_pose_model.pth")
weight_Path = '/media/wow/disk2/STT2/STTran-main/GCN/st_gcn.kinetics.pt'
first_para = 'GCN.st_gcn.Model'
second_para = {'in_channels': 3, 'num_class': 400, 'edge_importance_weighting': True,'graph_args': {'layout': 'openpose', 'strategy': 'spatial'}}


def extract(openpose_index,openpose_index_len,openposed_value_scale):
    pose_tracker = naive_pose_tracker(data_frame=openpose_index_len)
    frame_index = 0
    print(len(openpose_index))
    for i in range(len(openpose_index)):
        image = cv2.imread(root_path + openpose_index[i])
        image=cv2.resize(image,None,None,fx=float(openposed_value_scale),fy=float(openposed_value_scale),interpolation=cv2.INTER_LINEAR)
        w,h,_=image.shape
        candidate, subset = body_estimate(image)
        if len(subset)==0:
            continue
        multi_pose, posed_index, posed_index_not = estimate_bodypose(candidate, subset)

        multi_pose = torch.from_numpy(multi_pose)
        multi_pose = multi_pose.unsqueeze(0)

        multi_pose[:, :, 0] = multi_pose[:, :, 0] / w  # 横坐标
        multi_pose[:, :, 1] = multi_pose[:, :, 1] / h  # 纵坐标
        multi_pose[:, :, 0:2] = multi_pose[:, :, 0:2] - 0.5
        multi_pose[:, :, 0][multi_pose[:, :, 2] == 0] = 0
        multi_pose[:, :, 1][multi_pose[:, :, 2] == 0] = 0

        multi_pose = multi_pose.numpy()
        pose_tracker.update(multi_pose, frame_index)
        frame_index += 1

    data_numpy = pose_tracker.get_skeleton_sequence()
    data = torch.from_numpy(data_numpy)
    data = data.unsqueeze(0)
    data = data.float().to("cuda:0").detach()
    model = load_model(first_para, **second_para)
    model = load_weights(model, weight_Path, None)
    output, feature = model.extract_feature(data)
    print(output.shape)
    print(feature.shape)

    return feature




def load_model(model, **model_args):
        #model=GCN.st_gcn.Model
    Model = import_class(model)
    model = Model(**model_args)
    return model


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' %
                          (class_str,
                           traceback.format_exception(*sys.exc_info())))


def load_weights(model,weights,ignore_weights):
    model = load(model,weights,ignore_weights).cuda()
    return model


def load(model, weights_path, ignore_weights=None):
    if ignore_weights is None:
        ignore_weights = []
    if isinstance(ignore_weights, str):
        ignore_weights = [ignore_weights]

    weights = torch.load(weights_path)
    weights = OrderedDict([[k.split('module.')[-1],v.cpu()] for k, v in weights.items()])

    try:
        model.load_state_dict(weights)
    except (KeyError, RuntimeError):
        state = model.state_dict()
        diff = list(set(state.keys()).difference(set(weights.keys())))
        state.update(weights)
        model.load_state_dict(state)
    return model


class naive_pose_tracker():


    def __init__(self, data_frame=1, num_joint=18, max_frame_dis=np.inf):
        self.data_frame = data_frame#1
        self.num_joint = 18
        self.max_frame_dis = max_frame_dis
        self.latest_frame = 0
        self.trace_info = list()

    def update(self, multi_pose, current_frame):#(self,multi_pose,0)
        # multi_pose.shape: (num_person, num_joint, 3)

        if current_frame <= self.latest_frame:#1---->0<0       2------->1>0
            return

        if len(multi_pose.shape) != 3:
            print('multi_pose.shape error')
            return

        score_order = (-multi_pose[:, :, 2].sum(axis=1)).argsort(axis=0)#返回由小到大排序后index
        #对于sgg，score_order始终为0
        for p in multi_pose[score_order]:
            #p是每个坐标关节点
            # match existing traces
            matching_trace = None
            matching_dis = None
            for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
                # trace.shape: (num_frame, num_joint, 3)
                if current_frame <= latest_frame:
                    continue
                mean_dis, is_close = self.get_dis(trace, p)
                if is_close:
                    if matching_trace is None:
                        matching_trace = trace_index
                        matching_dis = mean_dis
                    elif matching_dis > mean_dis:
                        matching_trace = trace_index
                        matching_dis = mean_dis

            # update trace information
            if matching_trace is not None:
                trace, latest_frame = self.trace_info[matching_trace]

                # padding zero if the trace is fractured
                pad_mode = 'interp' if latest_frame == self.latest_frame else 'zero'
                pad = current_frame-latest_frame-1
                new_trace = self.cat_pose(trace, p, pad, pad_mode)
                self.trace_info[matching_trace] = (new_trace, current_frame)

            else:
                new_trace = np.array([p])
                #直接把第一个人输入
                self.trace_info.append((new_trace, current_frame))

        self.latest_frame = current_frame#1

    def get_skeleton_sequence(self):

        # remove old traces
        valid_trace_index = []
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            #还是第一个人的姿态
            if self.latest_frame - latest_frame < self.data_frame:
                valid_trace_index.append(trace_index)
        self.trace_info = [self.trace_info[v] for v in valid_trace_index]

        num_trace = len(self.trace_info)
        if num_trace == 0:
            return None

        data = np.zeros((3, self.data_frame, self.num_joint, num_trace))
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            end = self.data_frame - (self.latest_frame - latest_frame)
            d = trace[-end:]
            beg = end - len(d)
            data[:, beg:end, :, trace_index] = d.transpose((2, 0, 1))

        return data

    # concatenate pose to a trace
    def cat_pose(self, trace, pose, pad, pad_mode):
        # trace.shape: (num_frame, num_joint, 3)
        num_joint = pose.shape[0]
        num_channel = pose.shape[1]
        if pad != 0:
            if pad_mode == 'zero':
                trace = np.concatenate(
                    (trace, np.zeros((pad, num_joint, 3))), 0)
            elif pad_mode == 'interp':
                last_pose = trace[-1]
                coeff = [(p+1)/(pad+1) for p in range(pad)]
                interp_pose = [(1-c)*last_pose + c*pose for c in coeff]
                trace = np.concatenate((trace, interp_pose), 0)
        new_trace = np.concatenate((trace, [pose]), 0)
        return new_trace

    # calculate the distance between a existing trace and the input pose

    def get_dis(self, trace, pose):
        last_pose_xy = trace[-1, :, 0:2]
        curr_pose_xy = pose[:, 0:2]

        mean_dis = ((((last_pose_xy - curr_pose_xy)**2).sum(1))**0.5).mean()
        wh = last_pose_xy.max(0) - last_pose_xy.min(0)
        scale = (wh[0] * wh[1]) ** 0.5 + 0.0001
        is_close = mean_dis < scale * self.max_frame_dis
        return mean_dis, is_close

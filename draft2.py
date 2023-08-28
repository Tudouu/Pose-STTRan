import torch
import numpy as np
import cv2
from openpose.src.body import Body
from openpose.src.estimate_pose import estimate_bodypose
import pickle
import re

# k='a ssinfrontss b double'
# a=re.findall('infronts.*?double',k)
# print(a)

path = '/media/wow/disk2/AG/dataset/annotations/object_bbox_and_relationship.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径

f = open(path, 'rb')
data = pickle.load(f)
#print(type(data)) <class 'dict'>
print(data)
exit()
# print(data['0KER3.mp4/000243.png'])
# [{'class': 'book', 'bbox': (238.22180451127807, 203.55263157894734, 29.21052631578945, 7.631578947368439),
#   'attention_relationship': ['looking_at'], 'spatial_relationship': ['in_front_of'],
#   'contacting_relationship': ['not_contacting'],
#   'metadata': {'tag': '0KER3.mp4/book/000243', 'set': 'train'},
#   'visible': True},
#  {'class': 'box', 'bbox': (191.76442307692287, 174.96794871794867, 112.5, 66.87499999999997),
#   'attention_relationship': ['looking_at'], 'spatial_relationship': ['in_front_of'],
#   'contacting_relationship': ['touching'],
#   'metadata': {'tag': '0KER3.mp4/box/000243', 'set': 'train'},
#   'visible': True},
#  {'class': 'chair', 'bbox': None,
#   'attention_relationship': None, 'spatial_relationship': None,
#   'contacting_relationship': None,
#   'metadata': {'tag': '0KER3.mp4/chair/000243', 'set': 'train'},
#   'visible': False},
#  {'class': 'phone/camera', 'bbox': (327.359816653934, 71.65339317545201, 17.380952380952408, 17.14285714285714),
#   'attention_relationship': ['looking_at'], 'spatial_relationship': ['in_front_of'],
#   'contacting_relationship': ['holding'],
#   'metadata': {'tag': '0KER3.mp4/phone_camera/000243', 'set': 'train'},
#   'visible': True}]

#print(data)
exit()





















root_path='/media/wow/disk2/AG/dataset/frames/'
body_estimate=Body("openpose/model/body_pose_model.pth")

with open("/media/wow/disk2/AG/dataset/annotations/person_bbox.pkl", 'rb') as f:
    person_bbox = pickle.load(f)

print(person_bbox["99WON.mp4/000177.png"])

exit()
aaa=np.random.randint(1,5,(2,3,5,1))
intensity = (aaa * aaa).sum(axis=0)
print(aaa)
print('695')
print(aaa*aaa)
print((aaa*aaa).shape)

print('6546')
print(intensity)
print(intensity.shape)
exit()

class naive_pose_tracker():
    """ A simple tracker for recording person poses and generating skeleton sequences.
    For actual occasion, I recommend you to implement a robuster tracker.
    Pull-requests are welcomed.
    """

    def __init__(self, data_frame=128, num_joint=18, max_frame_dis=np.inf):
        self.data_frame = data_frame#1
        self.num_joint =4#18
        self.max_frame_dis = max_frame_dis#inf
        self.latest_frame = 0
        self.trace_info = list()

    def update(self, multi_pose, current_frame):
        #(,,0)
        # multi_pose.shape: (num_person, num_joint, 3)
        #第一张图片：0=0,直接return
        #第二张：current_frame=1!<self.latest_frame=0
        if current_frame <= self.latest_frame:
            print('----------1a',current_frame)
            return

        if len(multi_pose.shape) != 3:
            print('1b')
            return

        score_order = (-multi_pose[:, :, 2].sum(axis=1)).argsort(axis=0)#返回由小到大排序后index
        #对于sgg，score_order始终为0
        for p in multi_pose[score_order]:
            #p是每个关节坐标点
            # match existing traces
            matching_trace = None
            matching_dis = None

            for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
                # trace.shape: (num_frame, num_joint, 3)
                #print(current_frame)
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

        self.latest_frame = current_frame#0

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

        #self.data_frame=339 TODO change here
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


openpose_index=['8KM5Q.mp4/000008.png', '8KM5Q.mp4/000022.png', '8KM5Q.mp4/000034.png', '8KM5Q.mp4/000037.png', '8KM5Q.mp4/000051.png', '8KM5Q.mp4/000060.png', '8KM5Q.mp4/000065.png', '8KM5Q.mp4/000075.png', '8KM5Q.mp4/000087.png', '8KM5Q.mp4/000113.png', '8KM5Q.mp4/000139.png', '8KM5Q.mp4/000222.png', '8KM5Q.mp4/000369.png', '8KM5Q.mp4/000516.png', '8KM5Q.mp4/000663.png']
openpose_index_len=len(openpose_index)

pose_tracker = naive_pose_tracker(data_frame=openpose_index_len)
frame_index = 0
for i in range(len(openpose_index)):
    image = cv2.imread(root_path + openpose_index[i])
    image = cv2.resize(image, None, None, fx=float(1.6667), fy=float(1.6667),
                       interpolation=cv2.INTER_LINEAR)
    w, h, _ = image.shape
    candidate, subset = body_estimate(image)
    if len(subset) == 0:
        print('???')
        continue
    multi_pose, posed_index, posed_index_not = estimate_bodypose(candidate, subset)
    multi_pose=multi_pose[:4,:]
    print(i,multi_pose)
    multi_pose = torch.from_numpy(multi_pose)
    multi_pose = multi_pose.unsqueeze(0)

    #multi_pose[:, :, 0] = multi_pose[:, :, 0] / w  # 横坐标
    #multi_pose[:, :, 1] = multi_pose[:, :, 1] / h  # 纵坐标
    #multi_pose[:, :, 0:2] = multi_pose[:, :, 0:2] - 0.5
    #multi_pose[:, :, 0][multi_pose[:, :, 2] == 0] = 0
    #multi_pose[:, :, 1][multi_pose[:, :, 2] == 0] = 0

    multi_pose = multi_pose.numpy()

    pose_tracker.update(multi_pose, frame_index)
    frame_index += 1
    # if frame_index==2:
    # break

data_numpy = pose_tracker.get_skeleton_sequence()
data = torch.from_numpy(data_numpy)
data = data.unsqueeze(0)
np.set_printoptions(suppress=True)
#print(all)
#print(data.shape)
#print(data)
N, C, T, V, M = data.size()
print('******************************************')
x = data.permute(0, 4, 3, 1, 2).contiguous()
#print(x)
#print(x.shape)
x = x.view(N * M, V * C, T)
print('--------------------------------------------')
#print(x)
#print(x.shape)
x = x.view(N, M, V, C, T)
x = x.view(N * M, C, T, V)
print('///////////////////////////////')
print(x.shape)
print(x)
exit()

















multi_pose=np.random.randint(0,350,(18,2))
print(multi_pose)
exit()
multi_pose=np.array([[215,204,0.6],
                     [150,252,0.89],
                     [256,291,0.45],
                     [ 31,91,0.69],
                     [325,62,0.74],
                     [  2,57,0.66],
                     [159,78,0.841],
                     [ 75,180,0.46],
                     [326,98,0.49],
                     [284,125,0.58],
                     [307,338,0.91],
                     [139,129,0.48],
                     [235,131,0.56],
                     [113,257,0.84],
                     [143,131,0.68],
                     [ 17,338,0.48],
                     [320,83,0.75],
                     [ 96,234,0.47]])


pose_tracker = naive_pose_tracker(data_frame=10)
frame_index = 0
video=list()
while (True):
    multi_pose=np.random.randint()
    #print(multi_pose)
    #print(posed_index_not)
    #print(posed_index)
    #cv2.circle(orig_image,(73,71),1,(255,0,0),-1)
    #cv2.imshow('image',orig_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    multi_pose=torch.from_numpy(multi_pose)
    multi_pose=multi_pose.unsqueeze(0)

    multi_pose[:, :, 0] = multi_pose[:, :, 0] / W  # 横坐标
    multi_pose[:, :, 1] = multi_pose[:, :, 1] / H  # 纵坐标
    multi_pose[:, :, 0:2] = multi_pose[:, :, 0:2] - 0.5
    multi_pose[:, :, 0][multi_pose[:, :, 2] == 0] = 0
    multi_pose[:, :, 1][multi_pose[:, :, 2] == 0] = 0

    multi_pose=multi_pose.numpy()
    pose_tracker.update(multi_pose, frame_index)
    frame_index += 1
   # print('Pose estimation ({}/{}).'.format(frame_index))
   # if frame_index==2:
   #     break
print(np.shape(video))
print('-----------------------------------------')
data_numpy = pose_tracker.get_skeleton_sequence()

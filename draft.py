import numpy as np
#import torch
#import collections
#import cv2
import json
#from light_openpose.models.with_mobilenet import PoseEstimationWithMobileNet
#from light_openpose.modules.load_state import load_state
#from light_openpose import light_op
import re
#from stanfordcorenlp import StanfordCoreNLP


#nlp=StanfordCoreNLP(r'/media/wow/disk2/AG/stanford-corenlp-4.3.2')
#str='A woman takes a picture of another woman wearing a flower lei.'
# res=nlp.dependency_parse(str)
# res=nlp.parse('A woman takes a picture of another woman wearing a flower lei.')
# print(res)
# exit()

#json_path='/media/wow/disk2/AG/coco_anno/captions_val2017.json'#25014
#json_path='/media/wow/disk2/AG/coco_anno/captions_train2017.json'#591753
json_path=r'E:\STTran-latest\STTran-main\captions_val2017.json'
#caption_path='/media/wow/disk2/AG/draft'
f=open(json_path)
data=json.load(f)
#print(data['annotations'][10]['caption'])
#print(len(data['annotations']))
strs=[]
list= []
for i in range(len(data['annotations'])):
    #print(data['annotations'][i]['caption'])
    #strs.append(data['annotations'][i]['caption'])
    #a=re.findall('infronts','a ssinfrontkss b')

    s=str(data['annotations'][i]['caption'])
    pattern = r"man"
    #pattern=r"mirror.*?woman.*?drink"
    a=re.findall(pattern,s)
    if len(a):
        list+=[data['annotations'][i]['caption']]
        print(data['annotations'][i]['caption'],data['annotations'][i]['image_id'])
print(list)
exit()
textf = open(caption_path, 'w')
textf.write(str(strs))
textf.close()
exit()

a=torch.tensor([[5,4,7],
            [1,3,5],
            [4,5,5],
            [7,2,2],
            [1,1,6]])
print(a[2,0])
b=torch.argmax(a[3,0])
print(b)
exit()

pose=torch.tensor(np.array([[[193,  71,   1.   ],#鼻子
        [198,  67,   1.   ],#左眼
        [191,  67,   1.   ],#右眼
        [211 ,  70,   1.   ],#左耳
        [187,  71,   1.   ],#右耳
        [229, 105,   1.   ],#左肩
        [186, 103,   1.   ],#右肩
        [238, 146,   1.   ],#左肘关节
        [163, 123,   1.   ],#右肘关节
        [236, 184,   1.   ]]]))
pose = pose.permute(2, 0, 1)
x = pose.view(3, 1, 10, 1)
print(pose)
print(x)
exit()


i=0
fre_ske_idx=[]
openpose_rel_man_img_idx=[0,0,0,0,0,5,8,9,9,10,10,11]
while i < len(openpose_rel_man_img_idx):
    if i == len(openpose_rel_man_img_idx) - 1:
        fre_ske_idx.append(1)
        break
    num = 1
    while (i + 1) < len(openpose_rel_man_img_idx) and openpose_rel_man_img_idx[i] == openpose_rel_man_img_idx[i + 1]:
        num = num + 1
        if (i + 1) < len(openpose_rel_man_img_idx):
            i = i + 1
        else:
            break
    if i == len(openpose_rel_man_img_idx) - 1:
        break
    fre_ske_idx.append(num)
    i = i + 1
fre_ske_idx = np.array(fre_ske_idx)
print(fre_ske_idx)
exit()

d=torch.tensor([])
a=torch.tensor([[0,2,4],
               [5,8,63],
                [8,5,4]])
for i in range(3):
    d=torch.stack((d,a[i,:]))
print(d)
exit()
pose=np.array([[[193,  71,   1.   ],#鼻子
        [198,  67,   1.   ],#左眼
        [191,  67,   1.   ],#右眼
        [211 ,  70,   1.   ],#左耳
        [187,  71,   1.   ],#右耳
        [229, 105,   1.   ],#左肩
        [186, 103,   1.   ],#右肩
        [238, 146,   1.   ],#左肘关节
        [163, 123,   1.   ],#右肘关节
        [236, 184,   1.   ],#左手腕
        [176 ,  94 ,   1.   ],#右手腕
        [220, 184,   1.   ],#左胯关节
        [189, 184,   1.   ],#右胯关节
        [221, 238,   1.   ],#左膝盖
        [189, 240,   1.   ],#右膝盖
        [224, 267,1.   ],#左脚踝
        [191, 267,1.]],
              [[193, 71, 1.],  # 鼻子
               [198, 67, 1.],  # 左眼
               [191, 67, 1.],  # 右眼
               [211, 70, 1.],  # 左耳
               [187, 71, 1.],  # 右耳
               [229, 105, 1.],  # 左肩
               [186, 103, 1.],  # 右肩
               [238, 146, 1.],  # 左肘关节
               [163, 123, 1.],  # 右肘关节
               [236, 184, 1.],  # 左手腕
               [176, 94, 1.],  # 右手腕
               [220, 184, 1.],  # 左胯关节
               [189, 184, 1.],  # 右胯关节
               [221, 238, 1.],  # 左膝盖
               [189, 240, 1.],  # 右膝盖
               [224, 267, 1.],  # 左脚踝
               [191, 267, 1.]]]#右脚踝
              )
pose=torch.from_numpy(pose)
rel_num, pose_point, xyz = pose.shape
pose = pose.permute(2, 0, 1)
pose = pose.view(xyz, rel_num, pose_point, 1)
print(pose)
exit()

path="/media/wow/disk2/AG/dataset/frames/VK0OU.mp4/000010.png"
image=cv2.imread(path)
image = cv2.resize(image, None, None, fx=float(1.8868), fy=float(1.8868),interpolation=cv2.INTER_LINEAR)
cv2.circle(image,(521,107),1,(0,255,0),5)
cv2.imshow('pic',image)
cv2.waitKey()
cv2.destroyAllWindows()


exit()



net = PoseEstimationWithMobileNet()
checkpoint = torch.load('/media/wow/disk2/STT2/STTran-main/light_openpose/checkpoint_iter_370000.pth', map_location='cpu')
load_state(net, checkpoint)
multi_pose=torch.ones((17,3))
print()
path="/media/wow/disk2/AG/dataset/frames/07RDG.mp4/000048.png"
image=cv2.imread(path)
image = cv2.resize(image, None, None, fx=float(1.5306), fy=float(1.5306),
                   interpolation=cv2.INTER_LINEAR)
multi_pose = light_op.estimate_pose(net, image)
if len(multi_pose)==0:
    print('it is ok')
print(multi_pose)
exit()

'''pose=np.array([[193,  71,   1.   ],#鼻子
        [198,  67,   1.   ],#左眼
        [191,  67,   1.   ],#右眼
        [211 ,  70,   1.   ],#左耳
        [187,  71,   1.   ],#右耳
        [229, 105,   1.   ],#左肩
        [186, 103,   1.   ],#右肩
        [238, 146,   1.   ],#左肘关节
        [163, 123,   1.   ],#右肘关节
        [236, 184,   1.   ],#左手腕
        [176 ,  94 ,   1.   ],#右手腕
        [220, 184,   1.   ],#左胯关节
        [189, 184,   1.   ],#右胯关节
        [221, 238,   1.   ],#左膝盖
        [189, 240,   1.   ],#右膝盖
        [224, 267,   1.   ],#左脚踝
        [191, 267,   1.   ]])#右脚踝'''

cv2.circle(image,(565,233),1,(0,255,0),5)
cv2.imshow('pic',image)
cv2.waitKey()
cv2.destroyAllWindows()
exit()























A = np.zeros((18, 18))
edge=[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11),
         (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (4, 3), (3, 2), (7, 6), (6, 5), (13, 12),
        (12, 11), (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
for i,j in edge:
    A[j, i] = 1
    A[i, j] = 1
#print(A)
transfer_mat = [np.linalg.matrix_power(A, d) for d in range(1 + 1)]
#print(transfer_mat)
#print(np.stack(transfer_mat)[0])
arrive_mat = (np.stack(transfer_mat) > 0)
#print(arrive_mat)
hop_dis = np.zeros((18, 18)) + np.inf
for d in range(1, -1, -1):
    hop_dis[arrive_mat[d]] = d
adjacency = np.zeros((18,18))
valid_hop = range(0,2,1)
print(hop_dis)
for hop in valid_hop:
    adjacency[hop_dis == hop] = 1

Dl = np.sum(adjacency, 0)
num_node = adjacency.shape[0]
Dn = np.zeros((num_node, num_node))
for i in range(num_node):
    if Dl[i] > 0:
        Dn[i, i] = Dl[i]**(-1)
AD = np.dot(A, Dn)
print(AD)
exit()








"""gt=[[{'person_bbox': np.array([[131.104,  67.112, 254.437, 264.468]])},
    {'class': 15, 'bbox':[  0.   ,   0.   , 163.611, 269.956],
     'attention_relationship':[1], 'spatial_relationship': [4],
     'contacting_relationship': [12], 'metadata': {'tag': 'HAA4O.mp4/doorway/000216', 'set': 'train'},
     'visible': True},
    {'class': 13, 'bbox':[  2.  ,   2.5 , 161.76, 267.5 ], 'attention_relationship':[1],
     'spatial_relationship': [4], 'contacting_relationship': [12],
     'metadata': {'tag': 'HAA4O.mp4/door/000216', 'set': 'train'}, 'visible': True}],
   [{'person_bbox':np.array([[129.913,  66.464, 254.652, 264.727]])},
    {'class': 15, 'bbox':[  0.   ,   0.   , 163.   , 269.935], 'attention_relationship':[1],
     'spatial_relationship': [4], 'contacting_relationship':[12],
     'metadata': {'tag': 'HAA4O.mp4/doorway/000217', 'set': 'train'}, 'visible': True},
    {'class': 13, 'bbox': [  1.   ,   2.   , 162.   , 269.918],
     'attention_relationship': [1], 'spatial_relationship':[4],
     'contacting_relationship':[12], 'metadata': {'tag': 'HAA4O.mp4/door/000217', 'set': 'train'}, 'visible': True}]]

for i,j in enumerate(gt):
    #j是所有的图片字典
    print(i)
    gt_boxes = np.zeros([len(j), 4])
    gt_boxes[0] = j[0]['person_bbox']
    print(gt_boxes)
    print('-------------------')
    gt_labels = np.zeros(len(j))
    gt_labels[0] = 1

    print('21',j[1:])
    for m, n in enumerate(j[1:]):
        # 提取所有的物品框
        gt_boxes[m + 1, :] = n['bbox']
        gt_labels[m + 1] = n['class']
    print(gt_boxes)"""

pred_box=np.array([[195.5,60,320,176],
                   [264,65,341,172],
                   [151,107,199,167],
                   [152,152,207,167],
                   [143,152,158,165]])
gt_box=np.array([[191,60,319,174],
                 [167,112,193,160]])

N=gt_box.shape[0]#gt框
K = pred_box.shape[0]#pred框
overlaps = np.zeros((N, K))
for k in range(K):  # K是行，就是pred的所有的框
    box_area = (
            (pred_box[k, 2] - pred_box[k, 0] + 1) *
            (pred_box[k, 3] - pred_box[k, 1] + 1)
    )#预测的面积
    for n in range(N):  # 也是行，是gt的框
        iw = (
                min(gt_box[n, 2], pred_box[k, 2]) -#x右上取小
                max(gt_box[n, 0], pred_box[k, 0]) + 1#x左下取大
        )#用确定这两个框是不是有相交处
        if iw > 0:
            ih = (
                    min(gt_box[n, 3], pred_box[k, 3]) -#y右上取小
                    max(gt_box[n, 1], pred_box[k, 1]) + 1#y左下取大
            )
            if ih > 0:
                ua = float(
                    (gt_box[n, 2] - gt_box[n, 0] + 1) *
                    (gt_box[n, 3] - gt_box[n, 1] + 1) +
                    box_area - iw * ih)
                overlaps[n, k] = iw * ih / ua
IOUs=overlaps
IOUs=np.random.randint(1,10,(5,4))
print(IOUs)
#n(行)代表第几个gtbox，k(列)代表第几个pred_box
#故几行几列就是第某个gtbox与第某个predbox的交并比
#print(overlaps)
detector_found_idx = []
gt_relations = []
supply_relations = []
candidates = []
for c in np.argsort(-IOUs[:, 1]):
    print(c)

exit()
if sum(IOUs[:, 0] > 0.5) > 0:  # assign_IOU_threshold=0.5
    candidate = IOUs[:, 0].argmax()  # [labels[np.where(IOUs_bool[:, m])[0]] == 1]
    detector_found_idx.append(candidate)
    gt_relations.append(n)
    candidates.append(candidate)
print(candidate)
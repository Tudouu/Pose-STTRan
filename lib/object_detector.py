import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import cv2
import os

from lib.funcs import assign_relations
from lib.draw_rectangles.draw_rectangles import draw_union_boxes
from fasterRCNN.lib.model.faster_rcnn.resnet import resnet
from fasterRCNN.lib.model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from fasterRCNN.lib.model.roi_layers import nms

class detector(nn.Module):

    '''first part: object detection (image/video)'''

    def __init__(self, train, object_classes, use_SUPPLY, mode='predcls'):
        super(detector, self).__init__()

        self.is_train = train
        self.use_SUPPLY = use_SUPPLY
        self.object_classes = object_classes#37
        # ['__background__', 'person', 'bag', 'bed', 'blanket', 'book',
        #               'box', 'broom', 'chair', 'closet/cabinet', 'clothes',
        #               'cup/glass/bottle', 'dish', 'door', 'doorknob', 'doorway',
        #               'floor', 'food', 'groceries', 'laptop', 'light', 'medicine',
        #               'mirror', 'paper/notebook', 'phone/camera', 'picture', 'pillow',
        #               'refrigerator', 'sandwich', 'shelf', 'shoe', 'sofa/couch',
        #               'table', 'television', 'towel', 'vacuum', 'window']
        self.mode = mode

        self.fasterRCNN = resnet(classes=self.object_classes, num_layers=101, pretrained=False, class_agnostic=False)
        self.fasterRCNN.create_architecture()
        checkpoint = torch.load('fasterRCNN/models/faster_rcnn_ag.pth')
        self.fasterRCNN.load_state_dict(checkpoint['model'])

        self.ROI_Align = copy.deepcopy(self.fasterRCNN.RCNN_roi_align)
        self.RCNN_Head = copy.deepcopy(self.fasterRCNN._head_to_tail)

    def forward(self, im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all):
        #img_data:对img_info转换后图像的张量信息,shape-->[38,3,1067,600]
        #img_info:原图像格式信息,print(im_info.shape)-->torch.size([38,3]),[1067.0000,   600.000,   2.2222]
        #gt_boxes:标记框,torch.Size([38, 1, 5]),38个1行5列向量,无值
        #num_boxes:每个图片内有几个框,torch.Size([38]),[0,0,0,0,0...],无值
        #gt_annotation:对应的文件夹的gt_annotations
        #im_all=None
        #mode:sgdet
        #print(gt_annotation)
        #[...,[{'person_bbox': array([[131.104, 67.112, 254.437, 264.468]], dtype=float32)},
         #{'class': 15, 'bbox': array([0., 0., 163.611, 269.956]), 'attention_relationship': tensor([1]),
          #'spatial_relationship': tensor([4]), 'contacting_relationship': tensor([12]),
          #'metadata': {'tag': 'HAA4O.mp4/doorway/000216', 'set': 'train'}, 'visible': True},
         #{'class': 13, 'bbox': array([2., 2.5, 161.76, 267.5]), 'attention_relationship': tensor([1]),
          #'spatial_relationship': tensor([4]), 'contacting_relationship': tensor([12]),
          #'metadata': {'tag': 'HAA4O.mp4/door/000216', 'set': 'train'}, 'visible': True}],...]
        if self.mode == 'sgdet':
            counter = 0
            counter_image = 0

            # create saved-bbox, labels, scores, features
            FINAL_BBOXES = torch.tensor([]).cuda(0)#bbox
            FINAL_LABELS = torch.tensor([], dtype=torch.int64).cuda(0)#labels
            FINAL_SCORES = torch.tensor([]).cuda(0)#scores
            FINAL_FEATURES = torch.tensor([]).cuda(0)#features
            FINAL_BASE_FEATURES = torch.tensor([]).cuda(0)

            #print('num',im_data.shape[0])
            while counter < im_data.shape[0]:
                #counter<38
                #10张10张图片的取，10张10张的处理
                if counter + 10 < im_data.shape[0]:#30<38
                    #10个10个取
                    inputs_data = im_data[counter:counter + 10]#im_data[0:10]
                    inputs_info = im_info[counter:counter + 10]#im_info[0:10]
                    inputs_gtboxes = gt_boxes[counter:counter + 10]#gt_boxes[0:10]
                    inputs_numboxes = num_boxes[counter:counter + 10]#num_boxes[0:10]

                else:
                    inputs_data = im_data[counter:]
                    inputs_info = im_info[counter:]
                    inputs_gtboxes = gt_boxes[counter:]
                    inputs_numboxes = num_boxes[counter:]
                #print(inputs_data.shape) torch.Size([10, 3, 1067, 600])

                rois, cls_prob, bbox_pred, base_feat, roi_features = self.fasterRCNN(inputs_data, inputs_info,
                                                                                     inputs_gtboxes, inputs_numboxes)
                #print('3.6465',base_feat.shape)
                # ROIS  torch.Size([10, 100, 5])
                # cls_prob  torch.Size([10, 100, 37])
                # bbox_pred  torch.Size([10, 100, 148])
                # base_feat  torch.Size([10, 1024, 67, 38])
                # pooled_feat  torch.Size([10, 100, 2048])
                SCORES = cls_prob.data
                #torch.Size([10, 100, 37])
                #100个框的37个类的概率
                boxes = rois.data[:, :, 1:5]#取1到5列，坐标
                # bbox regression (class specific)
                box_deltas = bbox_pred.data
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor([0.1, 0.1, 0.2, 0.2]).cuda(0) \
                             + torch.FloatTensor([0.0, 0.0, 0.0, 0.0]).cuda(0)  # the first is normalize std, the second is mean
                box_deltas = box_deltas.view(-1, rois.shape[1], 4 * len(self.object_classes))  # post_NMS_NTOP: 30
                #print(box_deltas.shape) torch.Size([10, 100, 148])
                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                #print(pred_boxes.shape) torch.Size([10, 100, 148])
                PRED_BOXES = clip_boxes(pred_boxes, im_info.data, 1)
                PRED_BOXES /= inputs_info[0, 2] # original bbox scale!!!!!!!!!!!!!!
                #print(PRED_BOXES.shape) torch.Size([10, 100, 148])

                #traverse frames
                for i in range(rois.shape[0]):#10张图片
                    # images in the batch
                    scores = SCORES[i]
                    #每张图片类别分数
                    pred_boxes = PRED_BOXES[i]

                    for j in range(1, len(self.object_classes)):
                        #1到37
                        # NMS according to obj categories
                        inds = torch.nonzero(scores[:, j] > 0.1).view(-1) #0.05 is score threshold
                        # if there is det
                        if inds.numel() > 0:
                            cls_scores = scores[:, j][inds]
                            _, order = torch.sort(cls_scores, 0, True)
                            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                            cls_dets = cls_dets[order]
                            keep = nms(cls_boxes[order, :], cls_scores[order], 0.4) # NMS threshold
                            cls_dets = cls_dets[keep.view(-1).long()]

                            if j == 1:
                                # for person we only keep the highest score for person!
                                final_bbox = cls_dets[0,0:4].unsqueeze(0)
                                final_score = cls_dets[0,4].unsqueeze(0)
                                final_labels = torch.tensor([j]).cuda(0)
                                final_features = roi_features[i, inds[order[keep][0]]].unsqueeze(0)

                            else:
                                final_bbox = cls_dets[:, 0:4]
                                final_score = cls_dets[:, 4]
                                final_labels = torch.tensor([j]).repeat(keep.shape[0]).cuda(0)
                                final_features = roi_features[i, inds[order[keep]]]

                            final_bbox = torch.cat((torch.tensor([[counter_image]], dtype=torch.float).repeat(final_bbox.shape[0], 1).cuda(0),final_bbox), 1)
                            FINAL_BBOXES = torch.cat((FINAL_BBOXES, final_bbox), 0)
                            FINAL_LABELS = torch.cat((FINAL_LABELS, final_labels), 0)
                            FINAL_SCORES = torch.cat((FINAL_SCORES, final_score), 0)
                            FINAL_FEATURES = torch.cat((FINAL_FEATURES, final_features), 0)
                    FINAL_BASE_FEATURES = torch.cat((FINAL_BASE_FEATURES, base_feat[i].unsqueeze(0)), 0)
                    counter_image += 1

                counter += 10
            FINAL_BBOXES = torch.clamp(FINAL_BBOXES, 0)
            prediction = {'FINAL_BBOXES': FINAL_BBOXES, 'FINAL_LABELS': FINAL_LABELS, 'FINAL_SCORES': FINAL_SCORES,
                          'FINAL_FEATURES': FINAL_FEATURES, 'FINAL_BASE_FEATURES': FINAL_BASE_FEATURES}
            #print('111111111111111111111111111111111')
            #print(FINAL_BBOXES.shape)  #[box_num, 5],第一列是图片的编号从0开始
            #print('final_bboxes',FINAL_BBOXES.shape)
            #print(FINAL_LABELS.shape)  #[label_num](就是boxnum)
            #print(FINAL_SCORES.shape)  #[score_num]
            #print(FINAL_FEATURES.shape)    #[box_num, 2048]
            #print(FINAL_BASE_FEATURES.shape) #[picture_num, 1024, 67, 38]
            #print('22222222222222222222222222222222')

            if self.is_train:

                DETECTOR_FOUND_IDX, GT_RELATIONS, SUPPLY_RELATIONS, assigned_labels = assign_relations(prediction, gt_annotation, assign_IOU_threshold=0.5)
                #print(DETECTOR_FOUND_IDX)  [[0, 1], [0, 1], [0, 1, 4, 2], [0, 1, 4, 2], [0, 1, 9], [0, 5], [0, 3]]，[0,1]里的值代表FINAL_BBOXES里FINAL_BBOXES[:,0]=0的(第一个照片的所有预测框)的第0和第1个框
                #DETECTOR_FOUND_IDX.shape=[图片数量，每张图片里的框数量不一样]
                #GT_RELATIONS+SUPPLY_RELATIONS直接等于gt_annotation
                #print(SUPPLY_RELATIONS)
                #[[], [], [{'class': 31, 'bbox': array([209., 173., 433., 285.]), 'attention_relationship': tensor([1]), 'spatial_relationship': tensor([1, 3]), 'contacting_relationship': tensor([10,  6]), 'metadata': {'tag': 'RJFT8.mp4/sofa_couch/000584', 'set': 'train'}, 'visible': True}], [], [], [], [], []]
                #如果这张图片的框全被预测，那么在SUPPLY_RELATIONS的对应位置就是[],否则就是具体的字典
                if self.use_SUPPLY:
                    # supply the unfounded gt boxes by detector into the scene graph generation training
                    FINAL_BBOXES_X = torch.tensor([]).cuda(0)
                    FINAL_LABELS_X = torch.tensor([], dtype=torch.int64).cuda(0)
                    FINAL_SCORES_X = torch.tensor([]).cuda(0)
                    FINAL_FEATURES_X = torch.tensor([]).cuda(0)
                    assigned_labels = torch.tensor(assigned_labels, dtype=torch.long).to(FINAL_BBOXES_X.device)

                    for i, j in enumerate(SUPPLY_RELATIONS):
                        if len(j) > 0:
                            #gt_annotation_video.append(gt_annotation_frame)
                            # gt_annotation_frame=[{'person_bbox': person_bbox['001YG.mp4/000089.png']['bbox']},
                            #                     {class,bbox,attention_relationship,...}]
                            unfound_gt_bboxes = torch.zeros([len(j), 5]).cuda(0)#第j张图片有几个框没有被pred出来
                            unfound_gt_classes = torch.zeros([len(j)], dtype=torch.int64).cuda(0)
                            one_scores = torch.ones([len(j)], dtype=torch.float32).cuda(0)  # probability
                            for m, n in enumerate(j):
                                # if person box is missing or objects
                                if 'bbox' in n.keys():
                                    unfound_gt_bboxes[m, 1:] = torch.tensor(n['bbox']) * im_info[i, 2]  # don't forget scaling!
                                    unfound_gt_classes[m] = n['class']
                                else:
                                    # here happens always that IOU <0.5 but not unfounded
                                    unfound_gt_bboxes[m, 1:] = torch.tensor(n['person_bbox']) * im_info[i, 2]  # don't forget scaling!
                                    unfound_gt_classes[m] = 1  # person class index

                            DETECTOR_FOUND_IDX[i] = list(np.concatenate((DETECTOR_FOUND_IDX[i],
                                                                         np.arange(start=int(sum(FINAL_BBOXES[:, 0] == i)),
                                                                             stop=int(sum(FINAL_BBOXES[:, 0] == i)) + len(SUPPLY_RELATIONS[i]))), axis=0).astype('int64'))
                            #FINAL_BBOXES[:,0]==i并不一定能把这张图所有的框检测出来
                            #这个步骤其实就是把没检测到框的作为最后一个塞回去
                            GT_RELATIONS[i].extend(SUPPLY_RELATIONS[i])#塞回去

                            # compute the features of unfound gt_boxes
                            pooled_feat = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES[i].unsqueeze(0),unfound_gt_bboxes.cuda(0))
                            pooled_feat = self.fasterRCNN._head_to_tail(pooled_feat)
                            #cls_prob = F.softmax(self.fasterRCNN.RCNN_cls_score(pooled_feat), 1)

                            unfound_gt_bboxes[:, 0] = i
                            unfound_gt_bboxes[:, 1:] = unfound_gt_bboxes[:, 1:] / im_info[i, 2]
                            FINAL_BBOXES_X = torch.cat((FINAL_BBOXES_X, FINAL_BBOXES[FINAL_BBOXES[:, 0] == i], unfound_gt_bboxes))
                            #实际上是把没pred出来的bbox给加进去，现在是真正完整的所有的bbox了，也是混合着正确的和多余的box的一个集合
                            FINAL_LABELS_X = torch.cat((FINAL_LABELS_X, assigned_labels[FINAL_BBOXES[:, 0] == i],unfound_gt_classes))  # final label is not gt!
                            FINAL_SCORES_X = torch.cat((FINAL_SCORES_X, FINAL_SCORES[FINAL_BBOXES[:, 0] == i], one_scores))
                            FINAL_FEATURES_X = torch.cat((FINAL_FEATURES_X, FINAL_FEATURES[FINAL_BBOXES[:, 0] == i], pooled_feat))
                        else:#j为空
                            FINAL_BBOXES_X = torch.cat((FINAL_BBOXES_X, FINAL_BBOXES[FINAL_BBOXES[:, 0] == i]))
                            FINAL_LABELS_X = torch.cat((FINAL_LABELS_X, assigned_labels[FINAL_BBOXES[:, 0] == i]))
                            FINAL_SCORES_X = torch.cat((FINAL_SCORES_X, FINAL_SCORES[FINAL_BBOXES[:, 0] == i]))
                            FINAL_FEATURES_X = torch.cat((FINAL_FEATURES_X, FINAL_FEATURES[FINAL_BBOXES[:, 0] == i]))

                FINAL_DISTRIBUTIONS = torch.softmax(self.fasterRCNN.RCNN_cls_score(FINAL_FEATURES_X)[:, 1:], dim=1)
                global_idx = torch.arange(start=0, end=FINAL_BBOXES_X.shape[0])  # all bbox indices[0,1,2,3...]
                im_idx = []  # which frame are the relations belong to
                pair = []
                a_rel = []
                s_rel = []
                c_rel = []
                openpose_img_idx=[]
                #print('DE',len(DETECTOR_FOUND_IDX))#图片数量
                #现在的DETECTOR_FOUND_IDX是完整的所有的正确的
                #[[0, 3], [0, 2], [0, 2], [0, 5], [0, 5], [0, 3], [0, 5], [0, 2], [0, 2], [0, 5], [0, 8], [0, 6, 1],[0, 8, 9], [0, 9, 1], [0, 6, 7]]
                for i, j in enumerate(DETECTOR_FOUND_IDX):
                    # print('gt',GT_RELATIONS)
                    #[...,[{'person_bbox': array([[180.666, 67.97, 277.425, 247.034]], dtype=float32)},
                    # {'class': 17, 'bbox': array([186.193, 95.721, 204.877, 112.037]), 'attention_relationship': tensor([0]),
                    #                          'spatial_relationship': tensor([2]), 'contacting_relationship': tensor([5, 3]),
                    #                         'metadata': {'tag': 'J7TT5.mp4/food/000036', 'set': 'train'}, 'visible': True},
                    # {'class': 16, 'bbox': array([115., 80.5, 322., 270.]), 'attention_relationship': tensor([1]),
                    #                         'spatial_relationship': tensor([3, 1]), 'contacting_relationship': tensor([7]),
                    #                         'metadata': {'tag': 'J7TT5.mp4/floor/000036', 'set': 'train'}, 'visible': True}],...]

                    for k, kk in enumerate(GT_RELATIONS[i]):
                        #i代表图片的索引值
                        if 'person_bbox' in kk.keys():
                            kkk = k#k和kkk代表字典里一个个框的索引
                            break
                    localhuman = int(global_idx[FINAL_BBOXES_X[:, 0] == i][kkk])
                    #FINAL_BBOXES_X[:, 0] == i用来找第i个图片的所有框
                    #此处找到每个人的索引

                    for m, n in enumerate(j):
                        if 'class' in GT_RELATIONS[i][m].keys():
                            im_idx.append(i)

                            pair.append([localhuman, int(global_idx[FINAL_BBOXES_X[:, 0] == i][int(n)])])

                            a_rel.append(GT_RELATIONS[i][m]['attention_relationship'].tolist())
                            s_rel.append(GT_RELATIONS[i][m]['spatial_relationship'].tolist())
                            c_rel.append(GT_RELATIONS[i][m]['contacting_relationship'].tolist())

                pair = torch.tensor(pair).cuda(0)
                im_idx = torch.tensor(im_idx, dtype=torch.float).cuda(0)

                for i in range(pair.shape[0]):
                    openpose_img_idx.append(int(pair[i][0]))
                openpose_img_idx = np.array(openpose_img_idx)

                #print('opidx',np.array(openpose_img_idx).shape)#[关系数量]
                #print('im_idx',im_idx.shape) #[关系数量]
                #print('final_bboxes_x',FINAL_BBOXES_X.shape)
                #print('pair',pair.shape)#[关系数量,关系对]
                #print(pair)这些值都是global_idx里的索引号，这个索引真正对应的bbox坐标在FINAL_BBOXES_X里
                #[[0, 7],
                 #[0, 3],
                 #[11, 16],
                 #[11, 13],
                 #[18, 24],
                 #[18, 19],
                 #[26, 35],
                 #[26, 30],
                 #[38, 45],
                 #[38, 48],
                 #[49, 57],
                 #[49, 50]]

                #print('052',(torch.min(FINAL_BBOXES_X[:, 1:3][pair[:, 0]],FINAL_BBOXES_X[:, 1:3][pair[:, 1]])).shape)  #[关系数量,candicate]
                #print('054',(torch.max(FINAL_BBOXES_X[:, 3:5][pair[:, 0]],FINAL_BBOXES_X[:, 3:5][pair[:, 1]])).shape)  #[关系数量,candicate]
                union_boxes = torch.cat((im_idx[:, None],
                                         torch.min(FINAL_BBOXES_X[:, 1:3][pair[:, 0]],
                                                   FINAL_BBOXES_X[:, 1:3][pair[:, 1]]),
                                         torch.max(FINAL_BBOXES_X[:, 3:5][pair[:, 0]],
                                                   FINAL_BBOXES_X[:, 3:5][pair[:, 1]])), 1)
                #print('union_box',union_boxes.shape) [关系数, 5]
                union_boxes[:, 1:] = union_boxes[:, 1:] * im_info[0, 2]
                union_feat = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES, union_boxes)
                #print('union_feature',union_feat.shape)  #[关系数量，1024,7,7]


                pair_rois = torch.cat((FINAL_BBOXES_X[pair[:,0],1:],FINAL_BBOXES_X[pair[:,1],1:]), 1).data.cpu().numpy()
                spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(FINAL_FEATURES.device)

                entry = {'boxes': FINAL_BBOXES_X,
                         'labels': FINAL_LABELS_X,
                         'scores': FINAL_SCORES_X,
                         'distribution': FINAL_DISTRIBUTIONS,
                         'im_idx': im_idx,
                         'pair_idx': pair,
                         'features': FINAL_FEATURES_X,
                         'union_feat': union_feat,
                         'spatial_masks': spatial_masks,
                         'attention_gt': a_rel,
                         'spatial_gt': s_rel,
                         'contacting_gt': c_rel,
                         'openpose_img_idx':openpose_img_idx}

                return entry

            else:
                FINAL_DISTRIBUTIONS = torch.softmax(self.fasterRCNN.RCNN_cls_score(FINAL_FEATURES)[:, 1:], dim=1)
                FINAL_SCORES, PRED_LABELS = torch.max(FINAL_DISTRIBUTIONS, dim=1)
                PRED_LABELS = PRED_LABELS + 1
                #print(FINAL_DISTRIBUTIONS.shape) torch.Size([322, 36])
                #print(FINAL_SCORES.shape)  torch.Size([322])
                #print(FINAL_LABELS.shape)  torch.Size([322])

                entry = {'boxes': FINAL_BBOXES,#空间特征
                         'scores': FINAL_SCORES,
                         'distribution': FINAL_DISTRIBUTIONS,
                         'pred_labels': PRED_LABELS,
                         'features': FINAL_FEATURES,
                         'fmaps': FINAL_BASE_FEATURES,
                         'im_info': im_info[0, 2]}
                return entry
                #文件夹内所有图片特征、score、prob等都混在里一起
                #im_info[0, 2]=2.22
        #mode:predcle
        else:
            # how many bboxes we have
            bbox_num = 0

            im_idx = []  # which frame are the relations belong to
            pair = []
            a_rel = []
            s_rel = []
            c_rel = []

            for i in gt_annotation:
                bbox_num += len(i)
            #bbox_num=109
            #i=[{'person_bbox': array([[140.7502, 150.3986, 229.9369, 328.0558]], dtype=float32)},
            #   {'class': 32, 'bbox': array([208.2368, 229.0974, 270.    , 333.3916]),
            #               'attention_relationship': tensor([0]), 'spatial_relationship': tensor([2]),
            #               'contacting_relationship': tensor([8]),
            #               'metadata': {'tag': '00607.mp4/table/000016', 'set': 'test'},
            #               'visible': True}]
            #which means i=gt_annotation_frame
            FINAL_BBOXES = torch.zeros([bbox_num,5], dtype=torch.float32).cuda(0)
            FINAL_LABELS = torch.zeros([bbox_num], dtype=torch.int64).cuda(0)
            FINAL_SCORES = torch.ones([bbox_num], dtype=torch.float32).cuda(0)
            HUMAN_IDX = torch.zeros([len(gt_annotation),1], dtype=torch.int64).cuda(0)

            bbox_idx = 0
            for i, j in enumerate(gt_annotation):
                #i=0,1,2,3...index
                #j=[{'person_bbox'},{'class'...}]
                for m in j:
                    if 'person_bbox' in m.keys():
                        #m['person_bbox'][0]=[140.7502 150.3986 229.9369 328.0558]
                        FINAL_BBOXES[bbox_idx,1:] = torch.from_numpy(m['person_bbox'][0])
                        FINAL_BBOXES[bbox_idx, 0] = i
                        FINAL_LABELS[bbox_idx] = 1
                        HUMAN_IDX[i] = bbox_idx
                        bbox_idx += 1
                    else:
                        FINAL_BBOXES[bbox_idx,1:] = torch.from_numpy(m['bbox'])
                        FINAL_BBOXES[bbox_idx, 0] = i
                        FINAL_LABELS[bbox_idx] = m['class']
                        im_idx.append(i)
                        pair.append([int(HUMAN_IDX[i]), bbox_idx])
                        a_rel.append(m['attention_relationship'].tolist())
                        s_rel.append(m['spatial_relationship'].tolist())
                        c_rel.append(m['contacting_relationship'].tolist())
                        bbox_idx += 1
            pair = torch.tensor(pair).cuda(0)
            im_idx = torch.tensor(im_idx, dtype=torch.float).cuda(0)

            counter = 0
            FINAL_BASE_FEATURES = torch.tensor([]).cuda(0)

            while counter < im_data.shape[0]:
                #compute 10 images in batch and  collect all frames data in the video
                if counter + 10 < im_data.shape[0]:
                    inputs_data = im_data[counter:counter + 10]
                else:
                    inputs_data = im_data[counter:]
                base_feat = self.fasterRCNN.RCNN_base(inputs_data)
                FINAL_BASE_FEATURES = torch.cat((FINAL_BASE_FEATURES, base_feat), 0)
                counter += 10

            FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] * im_info[0, 2]
            FINAL_FEATURES = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES, FINAL_BBOXES)
            FINAL_FEATURES = self.fasterRCNN._head_to_tail(FINAL_FEATURES)

            if self.mode == 'predcls':

                union_boxes = torch.cat((im_idx[:, None], torch.min(FINAL_BBOXES[:, 1:3][pair[:, 0]], FINAL_BBOXES[:, 1:3][pair[:, 1]]),
                                         torch.max(FINAL_BBOXES[:, 3:5][pair[:, 0]], FINAL_BBOXES[:, 3:5][pair[:, 1]])), 1)
                union_feat = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES, union_boxes)
                FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] / im_info[0, 2]
                pair_rois = torch.cat((FINAL_BBOXES[pair[:, 0], 1:], FINAL_BBOXES[pair[:, 1], 1:]),
                                      1).data.cpu().numpy()
                spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(FINAL_FEATURES.device)

                entry = {'boxes': FINAL_BBOXES,
                         'labels': FINAL_LABELS, # here is the groundtruth
                         'scores': FINAL_SCORES,
                         'im_idx': im_idx,
                         'pair_idx': pair,
                         'human_idx': HUMAN_IDX,
                         'features': FINAL_FEATURES,
                         'union_feat': union_feat,
                         'union_box': union_boxes,
                         'spatial_masks': spatial_masks,
                         'attention_gt': a_rel,
                         'spatial_gt': s_rel,
                         'contacting_gt': c_rel
                        }

                return entry
            elif self.mode == 'sgcls':
                if self.is_train:

                    FINAL_DISTRIBUTIONS = torch.softmax(self.fasterRCNN.RCNN_cls_score(FINAL_FEATURES)[:, 1:], dim=1)
                    FINAL_SCORES, PRED_LABELS = torch.max(FINAL_DISTRIBUTIONS, dim=1)
                    PRED_LABELS = PRED_LABELS + 1

                    union_boxes = torch.cat(
                        (im_idx[:, None], torch.min(FINAL_BBOXES[:, 1:3][pair[:, 0]], FINAL_BBOXES[:, 1:3][pair[:, 1]]),
                         torch.max(FINAL_BBOXES[:, 3:5][pair[:, 0]], FINAL_BBOXES[:, 3:5][pair[:, 1]])), 1)
                    union_feat = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES, union_boxes)
                    FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] / im_info[0, 2]
                    pair_rois = torch.cat((FINAL_BBOXES[pair[:, 0], 1:], FINAL_BBOXES[pair[:, 1], 1:]),
                                          1).data.cpu().numpy()
                    spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(FINAL_FEATURES.device)

                    entry = {'boxes': FINAL_BBOXES,
                             'labels': FINAL_LABELS,  # here is the groundtruth
                             'scores': FINAL_SCORES,
                             'distribution': FINAL_DISTRIBUTIONS,
                             'pred_labels': PRED_LABELS,
                             'im_idx': im_idx,
                             'pair_idx': pair,
                             'human_idx': HUMAN_IDX,
                             'features': FINAL_FEATURES,
                             'union_feat': union_feat,
                             'union_box': union_boxes,
                             'spatial_masks': spatial_masks,
                             'attention_gt': a_rel,
                             'spatial_gt': s_rel,
                             'contacting_gt': c_rel}

                    return entry
                else:
                    FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] / im_info[0, 2]

                    FINAL_DISTRIBUTIONS = torch.softmax(self.fasterRCNN.RCNN_cls_score(FINAL_FEATURES)[:, 1:], dim=1)
                    FINAL_SCORES, PRED_LABELS = torch.max(FINAL_DISTRIBUTIONS, dim=1)
                    PRED_LABELS = PRED_LABELS + 1

                    entry = {'boxes': FINAL_BBOXES,
                             'labels': FINAL_LABELS,  # here is the groundtruth
                             'scores': FINAL_SCORES,
                             'distribution': FINAL_DISTRIBUTIONS,
                             'pred_labels': PRED_LABELS,
                             'im_idx': im_idx,
                             'pair_idx': pair,
                             'human_idx': HUMAN_IDX,
                             'features': FINAL_FEATURES,
                             'attention_gt': a_rel,
                             'spatial_gt': s_rel,
                             'contacting_gt': c_rel,
                             'fmaps': FINAL_BASE_FEATURES,
                             'im_info': im_info[0, 2]}

                    return entry

"""
Let's get the relationships yo
"""

import numpy as np
import torch
import torch.nn as nn

from lib.word_vectors import obj_edge_vectors
from lib.transformer import transformer
from lib.fpn.box_utils import center_size
from fasterRCNN.lib.model.roi_layers import ROIAlign, nms
from lib.draw_rectangles.draw_rectangles import draw_union_boxes
from GCN.model import Model


class ObjectClassifier(nn.Module):
    """
    Module for computing the object contexts and edge contexts
    """

    def __init__(self, mode='sgdet', obj_classes=None):
        super(ObjectClassifier, self).__init__()
        self.classes = obj_classes
        # ['__background__', 'person', 'bag', 'bed', 'blanket', 'book',
        #               'box', 'broom', 'chair', 'closet/cabinet', 'clothes',
        #               'cup/glass/bottle', 'dish', 'door', 'doorknob', 'doorway',
        #               'floor', 'food', 'groceries', 'laptop', 'light', 'medicine',
        #               'mirror', 'paper/notebook', 'phone/camera', 'picture', 'pillow',
        #               'refrigerator', 'sandwich', 'shelf', 'shoe', 'sofa/couch',
        #               'table', 'television', 'towel', 'vacuum', 'window']
        self.mode = mode#sgdet

        #----------add nms when sgdet
        self.nms_filter_duplicates = True
        self.max_per_img =64
        self.thresh = 0.01

        #roi align
        self.RCNN_roi_align = ROIAlign((7, 7), 1.0/16.0, 0)

        embed_vecs = obj_edge_vectors(obj_classes[1:], wv_type='glove.6B', wv_dir='data', wv_dim=200)
        self.obj_embed = nn.Embedding(len(obj_classes)-1, 200)
        self.obj_embed.weight.data = embed_vecs.clone()

        # This probably doesn't help it much
        self.pos_embed = nn.Sequential(nn.BatchNorm1d(4, momentum=0.01 / 10.0),
                                       nn.Linear(4, 128),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.1))
        self.obj_dim = 2048
        self.decoder_lin = nn.Sequential(nn.Linear(self.obj_dim + 200 + 128, 1024),
                                         nn.BatchNorm1d(1024),
                                         nn.ReLU(),
                                         nn.Linear(1024, len(self.classes)))

    def clean_class(self, entry, b, class_idx):
        final_boxes = []
        final_dists = []
        final_feats = []
        final_labels = []
        for i in range(b):
            scores = entry['distribution'][entry['boxes'][:, 0] == i]
            pred_boxes = entry['boxes'][entry['boxes'][:, 0] == i]
            feats = entry['features'][entry['boxes'][:, 0] == i]
            pred_labels = entry['pred_labels'][entry['boxes'][:, 0] == i]

            new_box = pred_boxes[entry['pred_labels'][entry['boxes'][:, 0] == i] == class_idx]
            new_feats = feats[entry['pred_labels'][entry['boxes'][:, 0] == i] == class_idx]
            new_scores = scores[entry['pred_labels'][entry['boxes'][:, 0] == i] == class_idx]
            new_scores[:, class_idx-1] = 0
            if new_scores.shape[0] > 0:
                new_labels = torch.argmax(new_scores, dim=1) + 1
            else:
                new_labels = torch.tensor([], dtype=torch.long).cuda(0)

            final_dists.append(scores)
            final_dists.append(new_scores)
            final_boxes.append(pred_boxes)
            final_boxes.append(new_box)
            final_feats.append(feats)
            final_feats.append(new_feats)
            final_labels.append(pred_labels)
            final_labels.append(new_labels)

        entry['boxes'] = torch.cat(final_boxes, dim=0)
        entry['distribution'] = torch.cat(final_dists, dim=0)
        entry['features'] = torch.cat(final_feats, dim=0)
        entry['pred_labels'] = torch.cat(final_labels, dim=0)
        return entry

    def forward(self, entry):
        #sgdet -->entry = {boxs, scores, distribution, pred_labels, features, fmaps, im_info}
        if self.mode  == 'predcls':
            entry['pred_labels'] = entry['labels']
            return entry
        elif self.mode == 'sgcls':

            obj_embed = entry['distribution'] @ self.obj_embed.weight
            pos_embed = self.pos_embed(center_size(entry['boxes'][:, 1:]))
            obj_features = torch.cat((entry['features'], obj_embed, pos_embed), 1)
            if self.training:
                entry['distribution'] = self.decoder_lin(obj_features)
                entry['pred_labels'] = entry['labels']
            else:
                entry['distribution'] = self.decoder_lin(obj_features)

                box_idx = entry['boxes'][:,0].long()
                b = int(box_idx[-1] + 1)

                entry['distribution'] = torch.softmax(entry['distribution'][:, 1:], dim=1)
                entry['pred_scores'], entry['pred_labels'] = torch.max(entry['distribution'][:, 1:], dim=1)
                entry['pred_labels'] = entry['pred_labels'] + 2

                # use the infered object labels for new pair idx
                HUMAN_IDX = torch.zeros([b, 1], dtype=torch.int64).to(obj_features.device)
                global_idx = torch.arange(0, entry['boxes'].shape[0])

                for i in range(b):
                    local_human_idx = torch.argmax(entry['distribution'][box_idx == i, 0]) # the local bbox index with highest human score in this frame
                    HUMAN_IDX[i] = global_idx[box_idx == i][local_human_idx]

                entry['pred_labels'][HUMAN_IDX.squeeze()] = 1
                entry['pred_scores'][HUMAN_IDX.squeeze()] = entry['distribution'][HUMAN_IDX.squeeze(), 0]

                # drop repeat overlap TODO!!!!!!!!!!!!
                for i in range(b):
                    duplicate_class = torch.mode(entry['pred_labels'][entry['boxes'][:, 0] == i])[0]
                    present = entry['boxes'][:, 0] == i
                    if torch.sum(entry['pred_labels'][entry['boxes'][:, 0] == i] ==duplicate_class) > 0:
                        duplicate_position = entry['pred_labels'][present] == duplicate_class

                        ppp = torch.argsort(entry['distribution'][present][duplicate_position][:,duplicate_class - 1])[:-1]
                        for j in ppp:

                            changed_idx = global_idx[present][duplicate_position][j]
                            entry['distribution'][changed_idx, duplicate_class-1] = 0
                            entry['pred_labels'][changed_idx] = torch.argmax(entry['distribution'][changed_idx])+1
                            entry['pred_scores'][changed_idx] = torch.max(entry['distribution'][changed_idx])


                im_idx = []  # which frame are the relations belong to
                pair = []
                for j, i in enumerate(HUMAN_IDX):
                    for m in global_idx[box_idx==j][entry['pred_labels'][box_idx==j] != 1]: # this long term contains the objects in the frame
                        im_idx.append(j)
                        pair.append([int(i), int(m)])

                pair = torch.tensor(pair).to(obj_features.device)
                im_idx = torch.tensor(im_idx, dtype=torch.float).to(obj_features.device)
                entry['pair_idx'] = pair
                entry['im_idx'] = im_idx

                entry['boxes'][:, 1:] = entry['boxes'][:, 1:] * entry['im_info']
                union_boxes = torch.cat((im_idx[:, None], torch.min(entry['boxes'][:, 1:3][pair[:, 0]], entry['boxes'][:, 1:3][pair[:, 1]]),
                                        torch.max(entry['boxes'][:, 3:5][pair[:, 0]], entry['boxes'][:, 3:5][pair[:, 1]])), 1)

                union_feat = self.RCNN_roi_align(entry['fmaps'], union_boxes)
                entry['boxes'][:, 1:] = entry['boxes'][:, 1:] / entry['im_info']
                pair_rois = torch.cat((entry['boxes'][pair[:, 0], 1:], entry['boxes'][pair[:, 1], 1:]),
                                      1).data.cpu().numpy()
                spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(obj_features.device)
                entry['union_feat'] = union_feat
                entry['union_box'] = union_boxes
                entry['spatial_masks'] = spatial_masks
            return entry
        else:
            if self.training:
                obj_embed = entry['distribution'] @ self.obj_embed.weight
                pos_embed = self.pos_embed(center_size(entry['boxes'][:, 1:]))
                obj_features = torch.cat((entry['features'], obj_embed, pos_embed), 1)

                box_idx = entry['boxes'][:, 0][entry['pair_idx'].unique()]
                l = torch.sum(box_idx == torch.mode(box_idx)[0])
                b = int(box_idx[-1] + 1)  # !!!

                entry['distribution'] = self.decoder_lin(obj_features)
                entry['pred_labels'] = entry['labels']
            else:

                obj_embed = entry['distribution'] @ self.obj_embed.weight#初始化
                pos_embed = self.pos_embed(center_size(entry['boxes'][:, 1:]))
                obj_features = torch.cat((entry['features'], obj_embed, pos_embed), 1) #use the result from FasterRCNN directly

                box_idx = entry['boxes'][:, 0].long()
                #找到每个框对应的图片的index
                b = int(box_idx[-1] + 1)#38

                entry = self.clean_class(entry, b, 5)
                entry = self.clean_class(entry, b, 8)
                entry = self.clean_class(entry, b, 17)

                # # NMS
                final_boxes = []
                final_dists = []
                final_feats = []
                for i in range(b):
                    # images in the batch
                    scores = entry['distribution'][entry['boxes'][:, 0] == i]
                    pred_boxes = entry['boxes'][entry['boxes'][:, 0] == i, 1:]
                    feats = entry['features'][entry['boxes'][:, 0] == i]

                    for j in range(len(self.classes) - 1):#36
                        # NMS according to obj categories
                        inds = torch.nonzero(torch.argmax(scores, dim=1) == j).view(-1)
                        # if there is det
                        if inds.numel() > 0:
                            cls_dists = scores[inds]
                            cls_feats = feats[inds]
                            cls_scores = cls_dists[:, j]
                            _, order = torch.sort(cls_scores, 0, True)
                            cls_boxes = pred_boxes[inds]
                            cls_dists = cls_dists[order]
                            cls_feats = cls_feats[order]
                            keep = nms(cls_boxes[order, :], cls_scores[order], 0.6)  # hyperparameter

                            final_dists.append(cls_dists[keep.view(-1).long()])
                            final_boxes.append(torch.cat((torch.tensor([[i]], dtype=torch.float).repeat(keep.shape[0],
                                                                                                        1).cuda(0),
                                                          cls_boxes[order, :][keep.view(-1).long()]), 1))
                            final_feats.append(cls_feats[keep.view(-1).long()])

                entry['boxes'] = torch.cat(final_boxes, dim=0)#最终
                #print(entry['boxes'].shape) torch.Size([278, 5])

                box_idx = entry['boxes'][:, 0].long()
                #更新index
                entry['distribution'] = torch.cat(final_dists, dim=0)#更新
                entry['features'] = torch.cat(final_feats, dim=0)#更新

                entry['pred_scores'], entry['pred_labels'] = torch.max(entry['distribution'][:, 1:], dim=1)
                entry['pred_labels'] = entry['pred_labels'] + 2

                # use the infered object labels for new pair idx
                HUMAN_IDX = torch.zeros([b, 1], dtype=torch.int64).to(box_idx.device)
                global_idx = torch.arange(0, entry['boxes'].shape[0])#(0,278)

                for i in range(b):
                    #i代表第几张图片
                    local_human_idx = torch.argmax(entry['distribution'][
                                                       box_idx == i, 0])  # the local bbox index with highest human score in this frame
                    HUMAN_IDX[i] = global_idx[box_idx == i][local_human_idx]

                entry['pred_labels'][HUMAN_IDX.squeeze()] = 1
                entry['pred_scores'][HUMAN_IDX.squeeze()] = entry['distribution'][HUMAN_IDX.squeeze(), 0]

                im_idx = []  # which frame are the relations belong to
                pair = []
                for j, i in enumerate(HUMAN_IDX):
                    for m in global_idx[box_idx == j][entry['pred_labels'][box_idx == j] != 1]:  # this long term contains the objects in the frame
                        im_idx.append(j)
                        pair.append([int(i), int(m)])

                pair = torch.tensor(pair).to(box_idx.device)
                im_idx = torch.tensor(im_idx, dtype=torch.float).to(box_idx.device)
                entry['pair_idx'] = pair
                entry['im_idx'] = im_idx
                entry['human_idx'] = HUMAN_IDX
                entry['boxes'][:, 1:] = entry['boxes'][:, 1:] * entry['im_info']
                union_boxes = torch.cat(
                    (im_idx[:, None], torch.min(entry['boxes'][:, 1:3][pair[:, 0]], entry['boxes'][:, 1:3][pair[:, 1]]),
                     torch.max(entry['boxes'][:, 3:5][pair[:, 0]], entry['boxes'][:, 3:5][pair[:, 1]])), 1)

                union_feat = self.RCNN_roi_align(entry['fmaps'], union_boxes)
                entry['boxes'][:, 1:] = entry['boxes'][:, 1:] / entry['im_info']
                entry['union_feat'] = union_feat
                entry['union_box'] = union_boxes
                pair_rois = torch.cat((entry['boxes'][pair[:, 0], 1:], entry['boxes'][pair[:, 1], 1:]),
                                      1).data.cpu().numpy()
                entry['spatial_masks'] = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(box_idx.device)
            return entry
            #entry={boxes-->框
                    #scores-->置信度
                    #distribution
                    #pred_labels-->预测的标签
                    #features
                    #fmaps
                    #im_info-->2.22
                    #pred_scores
                    #pair_idx
                    #im_idx#每个框所属的图片编号
                    #human_idx
                    #union_feat
                    #union_box
                    #spatial_masks}


class STTran(nn.Module):

    def __init__(self, mode='sgdet',
                 attention_class_num=None, spatial_class_num=None, contact_class_num=None, obj_classes=None, rel_classes=None,
                 enc_layer_num=None, dec_layer_num=None):
        #attention_class_num=3
        #spatial_class_num=6
        #contact_class_num=17
        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        """
        super(STTran, self).__init__()
        self.obj_classes = obj_classes
        # ['__background__', 'person', 'bag', 'bed', 'blanket', 'book',
        #               'box', 'broom', 'chair', 'closet/cabinet', 'clothes',
        #               'cup/glass/bottle', 'dish', 'door', 'doorknob', 'doorway',
        #               'floor', 'food', 'groceries', 'laptop', 'light', 'medicine',
        #               'mirror', 'paper/notebook', 'phone/camera', 'picture', 'pillow',
        #               'refrigerator', 'sandwich', 'shelf', 'shoe', 'sofa/couch',
        #               'table', 'television', 'towel', 'vacuum', 'window']
        self.rel_classes = rel_classes
        self.attention_class_num = attention_class_num#3
        self.spatial_class_num = spatial_class_num#6
        self.contact_class_num = contact_class_num#17
        assert mode in ('sgdet', 'sgcls', 'predcls')
        self.mode = mode

        self.object_classifier = ObjectClassifier(mode=self.mode, obj_classes=self.obj_classes)
        # entry={boxes-->框
        # scores-->置信度
        # distribution
        # pred_labels-->预测的标签
        # features
        # fmaps
        # im_info-->2.22
        # pred_scores
        # pair_idx
        # im_idx#每个框所属的图片编号
        # human_idx
        # union_feat
        # union_box
        # spatial_masks}
        ###################################
        self.union_func1 = nn.Conv2d(1024, 256, 1, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 256 //2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256//2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256 // 2, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, momentum=0.01),
        )
        self.pose_feature_extract=Model(in_channels=3,graph_args={'layout': 'openpose', 'strategy': 'spatial'},edge_importance_weighting=True)
        self.subj_fc = nn.Linear(2048, 512)
        self.obj_fc = nn.Linear(2048, 512)
        self.vr_fc = nn.Linear(256*7*7, 512)
        #self.pose_fc=nn.Linear(128*18*1,256)

        embed_vecs = obj_edge_vectors(obj_classes, wv_type='glove.6B', wv_dir='/media/jocker/disk2/AG', wv_dim=200)
        self.obj_embed = nn.Embedding(len(obj_classes), 200)
        self.obj_embed.weight.data = embed_vecs.clone()

        self.obj_embed2 = nn.Embedding(len(obj_classes), 200)
        self.obj_embed2.weight.data = embed_vecs.clone()

        #enc_layer_num=1
        #dec_layer_num=3
        self.glocal_transformer = transformer(enc_layer_num=enc_layer_num, dec_layer_num=dec_layer_num, embed_dim=1936, nhead=8,#1 3 1936 8
                                              dim_feedforward=2048, dropout=0.1, mode='latter')#2048 0.1 ‘latter’

        self.a_rel_compress = nn.Linear(1936, self.attention_class_num)#in_feature=(1936,1),out_feature=(3,1)TODO 1936变成2192
        self.s_rel_compress = nn.Linear(1936, self.spatial_class_num)#(1936,1),(6,1)   TODO 1936变成2192
        self.c_rel_compress = nn.Linear(1936, self.contact_class_num)#(1936,1),(17,1)  TODO 1936变成2192

    def pose_process(self,pose):
        rel_num, pose_point, xyz = pose.shape
        pose = pose.permute(2, 0, 1)
        x = pose.view(xyz, rel_num, pose_point, 1)  # (3,rel_num,18,1)

        # data normalization(3,rel_num,18,1)
        channel, rel_num, pose_num, l = x.shape
        x = x.permute(3, 2, 0, 1).contiguous()  # [l,pose_num,channel,rel_num]
        x = x.view(1, 54, rel_num)  # [l,pose_num*channel,rel_num]

        return x

    def forward(self,modee,entry,fre_ske_idx):
        #sgdet -->entry = {boxs, scores, distribution, pred_labels, features, fmaps, im_info}
        #sgdet train:{['boxes', 'labels', 'scores', 'distribution', 'im_idx', 'pair_idx', 'features', 'union_feat', 'spatial_masks', 'attention_gt', 'spatial_gt', 'contacting_gt']}

        entry = self.object_classifier(entry)
        #pair_idx:物体和人配对
        #数字代表第几个框
        #类似于：
        # [[0, 1],
        #  [0, 2],
        #  [0, 3],
        #  [0, 4],
        #  [0, 5],
        #  [0, 6],
        #  [0, 7],
        #  [0, 8],
        #  [0, 9],
        #  [10, 11],
        #  [10, 12],
        #  [10, 13]]


        #gcn提取特征部分
        # if not modee:
        #     i = 0
        #     openpose_img_idx = []
        #     fre_ske_idx = []
        #     pair=entry['pair_idx']
        #     #print(entry['pair_idx'])
        #     for j in range(pair.shape[0]):#229
        #         openpose_img_idx.append(int(pair[j][0]))
        #     openpose_rel_man_img_idx = np.array(openpose_img_idx)
        #     while i < len(openpose_rel_man_img_idx):
        #         if i == len(openpose_rel_man_img_idx) - 1:
        #             fre_ske_idx.append(1)
        #             break
        #         num = 1
        #         while (i+1)<len(openpose_rel_man_img_idx) and openpose_rel_man_img_idx[i] == openpose_rel_man_img_idx[i + 1]:
        #             num = num + 1
        #             if (i+1)<len(openpose_rel_man_img_idx):
        #                 i = i + 1
        #             else:
        #                 fre_ske_idx.append(num)
        #                 break
        #         if i == len(openpose_rel_man_img_idx) - 1:
        #             fre_ske_idx.append(num)
        #             break
        #         fre_ske_idx.append(num)
        #         i = i + 1
        #     fre_ske_idx = np.array(fre_ske_idx)

        #print('+++++++++++++++++++++++++++++')
        #print(entry['boxes'].shape)   torch.Size([278, 5])
        #entry={boxes,scores,distribution,pred_labels,features,fmaps,im_info,pred_scores,pair_idx,im_idx,
        #                                                     human_idx,union_feat,union_box,sptial_mask}
        # pose=torch.tensor([item.cpu().detach().numpy() for item in pose]).to(entry['features'].device)#[rel_num,18,3]
        # pose=self.pose_process(pose)
        # pose_feature=self.pose_feature_extract(pose,fre_ske_idx).contiguous().float()#[rel_num,128,18,1]
        # pose_rep=self.pose_fc(pose_feature.view(-1,128*18*1))#[rel_num,256]

        # visual part视觉部分
        #print(entry['pair_idx'])#包含所有的框、人对应关系

        subj_rep = entry['features'][entry['pair_idx'][:, 0]]
        #print('subj_rep', subj_rep.shape)
        subj_rep = self.subj_fc(subj_rep)#print('subj_rep',subj_rep.shape)[rel_num, 512]
        #print('subj_rep', subj_rep.shape)
        obj_rep = entry['features'][entry['pair_idx'][:, 1]]
        obj_rep = self.obj_fc(obj_rep)#print('obj_rep',obj_rep.shape)[rel_num, 512]

        #sptial part空间部分
        #print('entry['union_feat'].shape',entry['union_feat'].shape)[关系数量,1024,7,7]
        #print(entry['spatial_masks'].shape)[关系数量,2,27,27]
        #print((self.union_func1(entry['union_feat'])).shape) #[关系数量,256,7,7]
        #print((self.conv(entry['spatial_masks'])).shape)#[关系数量,256,7,7]
        #print('self.union_func1(entry['union_feat']',self.union_func1(entry['union_feat']))
        #print('self.conv(entry['spatial_masks']',self.conv(entry['spatial_masks']))
        vr = self.union_func1(entry['union_feat'])+self.conv(entry['spatial_masks'])#TODO here
        #print('vr1.shape',vr.shape)#[rel_num, 256, 7, 7]
        vr = self.vr_fc(vr.view(-1,256*7*7))#[rel_num, 512]
        #print('vr2.shape',vr.shape)

        #visual part和sptial part相连接
        x_visual = torch.cat((subj_rep, obj_rep, vr), 1)
        #print('x_visual.shape',x_visual.shape)#[rel_num, 512*3]

        # semantic part语义部分
        subj_class = entry['pred_labels'][entry['pair_idx'][:, 0]]
        obj_class = entry['pred_labels'][entry['pair_idx'][:, 1]]
        subj_emb = self.obj_embed(subj_class)
        obj_emb = self.obj_embed2(obj_class)
        #print(subj_emb.shape) 240*200
        #print(obj_emb.shape) 240*200
        x_semantic = torch.cat((subj_emb, obj_emb), 1)
        #print('x_semantic',x_semantic.shape)#[rel_num, 400]

        #四者连接-->relationship representation x 空间坐标特征+语义特征+视觉图像特征+姿态特征
        rel_features = torch.cat((x_visual, x_semantic), dim=1)
        #print('rel_features',rel_features.shape)#[rel_num, 1536+400]   现在[rel_num,1936+256=2192]
        # Spatial-Temporal Transformer

        global_output, global_attention_weights, local_attention_weights = self.glocal_transformer(features=rel_features, im_idx=entry['im_idx'])

        #print(global_output.shape)  torch.Size([240, 1936])
        #print(global_attention_weights.shape)  torch.Size([3, 37, 24, 24])
        #print(local_attention_weights.shape)  torch.Size([1, 38, 12, 12])
        entry["attention_distribution"] = self.a_rel_compress(global_output)#全连接层，分类
        entry["spatial_distribution"] = self.s_rel_compress(global_output)
        #print(entry['spatial_distribution'].size())-->(240,6)
        entry["contacting_distribution"] = self.c_rel_compress(global_output)

        entry["spatial_distribution"] = torch.sigmoid(entry["spatial_distribution"])
        entry["contacting_distribution"] = torch.sigmoid(entry["contacting_distribution"])

        #print(len(entry["boxes"])) 278
        #print(entry["human_idx"]) 人类框的index
        #print(len(entry["pair_idx"])) 240人和物体的配对
        #print(entry["im_idx"]):
        # [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1.,
        #  1., 2., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 3., 3.,
        #  4., 4., 4., 4., 4., 4., 4., 4., 4., 5., 5., 5., 5., 6.,
        #  6., 6., 6., 6., 6., 6., 6., 6., 6., 7., 7., 7., 7., 7.,
        #  7., 7., 8., 8., 8., 8., 8., 8., 8., 8., 9., 9., 9., 9.,
        #  9., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 11.,
        #  11., 11., 11., 11., 11., 11., 12., 12., 12., 12., 12., 12., 13., 13.,
        #  13., 13., 13., 13., 14., 14., 14., 15., 15., 15., 15., 15., 15., 16.,
        #  16., 16., 16., 16., 16., 17., 17., 17., 18., 18., 18., 18., 18., 18.,
        #  18., 18., 18., 19., 19., 19., 19., 19., 19., 19., 19., 19., 19., 20.,
        #  20., 20., 20., 20., 20., 20., 20., 21., 21., 21., 21., 21., 21., 21.,
        #  22., 22., 22., 22., 23., 23., 23., 23., 23., 23., 23., 24., 24., 24.,
        #  24., 24., 24., 25., 25., 25., 25., 25., 26., 26., 26., 26., 26., 27.,
        #  27., 27., 27., 27., 27., 27., 28., 28., 28., 28., 29., 29., 29., 29.,
        #  29., 30., 30., 30., 30., 30., 31., 31., 31., 31., 31., 31., 31., 31.,
        #  32., 32., 32., 32., 32., 32., 32., 33., 33., 33., 34., 34., 34., 35.,
        #  35., 35., 35., 35., 36., 36., 36., 36., 37., 37., 37., 37., 37., 37.,
        #  37., 37.], device = 'cuda:0')
        return entry
"""boxes
scores
distribution
pred_labels
features
fmaps
im_info
pred_scores
pair_idx
im_idx
human_idx
union_feat
union_box
spatial_masks
attention_distribution
spatial_distribution
contacting_distribution"""



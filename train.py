import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
np.set_printoptions(precision=3)
import os
import pandas as pd
import copy
import cv2
import time

from dataloader.action_genome import AG, cuda_collate_fn
from lib.object_detector import detector
from lib.config import Config
from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.AdamW import AdamW
from lib.sttran import STTran
from GCN import extract_feature
from light_openpose.extract_pose import extract

"""------------------------------------some settings----------------------------------------"""
conf = Config()
print('The CKPT saved here:', conf.save_path)
if not os.path.exists(conf.save_path):
    os.mkdir(conf.save_path)
print('spatial encoder layer num: {} / temporal decoder layer num: {}'.format(conf.enc_layer, conf.dec_layer))
for i in conf.args:
    print(i,':', conf.args[i])
"""-----------------------------------------------------------------------------------------"""

AG_dataset_train = AG(mode="train", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                      filter_small_box=False if conf.mode == 'predcls' else True)
dataloader_train = torch.utils.data.DataLoader(AG_dataset_train, shuffle=True, num_workers=4,
                                               collate_fn=cuda_collate_fn, pin_memory=False)
AG_dataset_test = AG(mode="test", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                     filter_small_box=False if conf.mode == 'predcls' else True)
dataloader_test = torch.utils.data.DataLoader(AG_dataset_test, shuffle=False, num_workers=4,
                                              collate_fn=cuda_collate_fn, pin_memory=False)


root_path="/media/jocker/disk2/AG/dataset/frames/"

gpu_device = torch.device("cuda:0")
# freeze the detection backbone
object_detector = detector(train=True, object_classes=AG_dataset_train.object_classes, use_SUPPLY=True, mode=conf.mode).to(device=gpu_device)
object_detector.eval()
model = STTran(mode=conf.mode,
               attention_class_num=len(AG_dataset_train.attention_relationships),
               spatial_class_num=len(AG_dataset_train.spatial_relationships),
               contact_class_num=len(AG_dataset_train.contacting_relationships),
               obj_classes=AG_dataset_train.object_classes,
               enc_layer_num=conf.enc_layer,
               dec_layer_num=conf.dec_layer).to(device=gpu_device)

evaluator =BasicSceneGraphEvaluator(mode=conf.mode,
                                    AG_object_classes=AG_dataset_train.object_classes,
                                    AG_all_predicates=AG_dataset_train.relationship_classes,
                                    AG_attention_predicates=AG_dataset_train.attention_relationships,
                                    AG_spatial_predicates=AG_dataset_train.spatial_relationships,
                                    AG_contacting_predicates=AG_dataset_train.contacting_relationships,
                                    iou_threshold=0.5,
                                    constraint='with')

#ckpt = torch.load('/media/wow/disk2/AG/save_protected/model_modified_                                                                                                                                   512T256_3.tar', map_location=gpu_device)
#model.load_state_dict(ckpt['state_dict'], strict=False)
#print('CKPT is loaded')

# loss function, default Multi-label margin loss
if conf.bce_loss:
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss()
else:
    ce_loss = nn.CrossEntropyLoss()
    mlm_loss = nn.MultiLabelMarginLoss()

# optimizer
if conf.optimizer == 'adamw':
    optimizer = AdamW(model.parameters(), lr=conf.lr)
elif conf.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=conf.lr)
elif conf.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=conf.lr, momentum=0.9, weight_decay=0.01)

scheduler = ReduceLROnPlateau(optimizer, "max", patience=1, factor=0.5, verbose=True, threshold=1e-4, threshold_mode="abs", min_lr=1e-7)

# some parameters
tr = []

epoch=1
all_loss = []
all_losses=[]
while epoch<=12:
    model.train()
    object_detector.train_x = True
    train_iter = iter(dataloader_train)
    test_iter = iter(dataloader_test)
    iter_train=1
    for b in range(len(dataloader_train)):#max_length=94,all_image=166785,ave=23
        #print("epoch {} iter_train {} is going to run".format(num,iter_train))
        data = next(train_iter)
#        return img_tensor, im_info, gt_boxes, num_boxes, index, openpose_frame,origin_img
        im_data = copy.deepcopy(data[0].cuda(0))
        im_info = copy.deepcopy(data[1].cuda(0))
        gt_boxes = copy.deepcopy(data[2].cuda(0))#全是0
        num_boxes = copy.deepcopy(data[3].cuda(0))#全是0
        gt_annotation = AG_dataset_train.gt_annotations[data[4]]#5771 具体的文件夹编号
        gt_pose=AG_dataset_train.gt_pose[data[4]]
        openpose_index=copy.deepcopy(data[5])#进行训练的图片
        img_h_l=copy.deepcopy(data[6])
        openpose_index_len=len(openpose_index)
        openposed_value_scale=im_info[0][2]


        # prevent gradients to FasterRCNN
        time_a=time.time()
        with torch.no_grad():
            entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation ,im_all=None)
        time_b=time.time()
        with torch.no_grad():
            pose,fre_ske_idx=extract(True,openpose_index,openpose_index_len,openposed_value_scale,entry['openpose_img_idx'],gt_pose)
        time_c=time.time()


        feature_graph=extract_feature.extract(pose,fre_ske_idx)
        pred = model(True,entry,pose,fre_ske_idx)
        pred=model(True,entry,fre_ske_idx)
        time_d=time.time()
        attention_distribution = pred["attention_distribution"]
        spatial_distribution = pred["spatial_distribution"]
        contact_distribution = pred["contacting_distribution"]

        attention_label = torch.tensor(pred["attention_gt"], dtype=torch.long).to(device=attention_distribution.device).squeeze()
        if not conf.bce_loss:
            # multi-label margin loss or adaptive loss
            spatial_label = -torch.ones([len(pred["spatial_gt"]), 6], dtype=torch.long).to(device=attention_distribution.device)
            contact_label = -torch.ones([len(pred["contacting_gt"]), 17], dtype=torch.long).to(device=attention_distribution.device)
            for i in range(len(pred["spatial_gt"])):
                spatial_label[i, : len(pred["spatial_gt"][i])] = torch.tensor(pred["spatial_gt"][i])
                contact_label[i, : len(pred["contacting_gt"][i])] = torch.tensor(pred["contacting_gt"][i])
        else:
            # bce loss
            spatial_label = torch.zeros([len(pred["spatial_gt"]), 6], dtype=torch.float32).to(device=attention_distribution.device)
            contact_label = torch.zeros([len(pred["contacting_gt"]), 17], dtype=torch.float32).to(device=attention_distribution.device)
            for i in range(len(pred["spatial_gt"])):
                spatial_label[i, pred["spatial_gt"][i]] = 1
                contact_label[i, pred["contacting_gt"][i]] = 1

        losses = {}
        if conf.mode == 'sgcls' or conf.mode == 'sgdet':
            losses['object_loss'] = ce_loss(pred['distribution'], pred['labels'])

        losses["attention_relation_loss"] = ce_loss(attention_distribution, attention_label)
        if not conf.bce_loss:
            losses["spatial_relation_loss"] = mlm_loss(spatial_distribution, spatial_label)
            losses["contact_relation_loss"] = mlm_loss(contact_distribution, contact_label)

        else:
            losses["spatial_relation_loss"] = bce_loss(spatial_distribution, spatial_label)
            losses["contact_relation_loss"] = bce_loss(contact_distribution, contact_label)

        optimizer.zero_grad()
        loss = sum(losses.values())
        loss.backward()
        all_loss.append(loss)
        all_losses.append(losses)
        print('epoch {} iter_train {}：picture_num {},loss {}'.format(epoch,iter_train,openpose_index_len,loss))
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
        optimizer.step()

        tr.append(pd.Series({x: y.item() for x, y in losses.items()}))
        iter_train=iter_train+1

        #if b % 1000 == 0 and b >= 1000:
         #   time_per_batch = (time.time() - start) / 1000
         #   print("\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, b, len(dataloader_train),
         #                                                                       time_per_batch, len(dataloader_train) * time_per_batch / 60))

 #           mn = pd.concat(tr[-1000:], axis=1).mean(1)
 #           print(mn)
 #           start =time.time()

    # eal_path='/media/wow/disk2/AG/loss/epoch_recall_512T256_delossT{}'.format(epoch)
    # all_loss_path='/media/wow/disk2/AG/loss/epoch_loss_512T256_delossT{}'.format(epoch)
    # all_losses_path='/media/wow/disk2/AG/loss/epoch_losses_512T256_delossT{}'.format(epoch)
    # with open(all_loss_path,'a',encoding='utf-8') as f:
    #     f.write(str(all_loss))
    #     f.close()
    # with open(all_losses_path,'a',encoding='utf-8') as f:
    #     f.write(str(all_losses))
    #     f.close()


    torch.save({"state_dict": model.state_dict()}, os.path.join(conf.save_path, "model_modified_512T256_{}.tar".format(epoch)))
    print("epoch_train {} has done".format(epoch))
    print("*" * 40)
    print("save the checkpoint {} epochs".format(epoch))

    model.eval()
    object_detector.is_train = False
    print("now is going to evaluate")
    iter_test=1
    with torch.no_grad():
        for b in range(len(dataloader_test)):
            print("epoch {} iter_test {} is going to run".format(epoch,iter_test))
            data = next(test_iter)
            im_data = copy.deepcopy(data[0].cuda(0))
            im_info = copy.deepcopy(data[1].cuda(0))
            gt_boxes = copy.deepcopy(data[2].cuda(0))
            num_boxes = copy.deepcopy(data[3].cuda(0))
            gt_annotation = AG_dataset_test.gt_annotations[data[4]]
            openpose_index = copy.deepcopy(data[5])
            img_h_l = copy.deepcopy(data[6])
            openpose_index_len = len(openpose_index)
            openposed_value_scale = im_info[0][2]


            entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
            pose = extract(False, openpose_index, openpose_index_len, openposed_value_scale, None, None)


            pred = model(False,entry,pose,None)
            evaluator.evaluate_scene_graph(gt_annotation, pred)
            iter_test=iter_test+1

    score = np.mean(evaluator.result_dict[conf.mode + "_recall"][20])
    evaluator.print_stats(epoch,eval_path)
    evaluator.reset_result()
    scheduler.step(score)
    print("evaluate {} done,the next epoch is going to run".format(epoch))

    epoch=epoch+1

    object_detector.is_train =True #!!!!!!!!!!!!!!!!!!!!! 原作者没写这行




    #PredCls：在有Bounding Box和Object Label的情况下对物体间Relation预测实验。
    #SGCls：在有Bounding Box的情况下，对Object Lable的预测和物体间Relation预测的实验。
    #SGGen：直接对Bounding Box，Object Label和物体间Relation预测的实验。

import numpy as np
np.set_printoptions(precision=4)
import copy
import torch
import time
from dataloader.action_genome import AG, cuda_collate_fn

from lib.config import Config
from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.object_detector import detector
from lib.sttran import STTran
from light_openpose.extract_pose import extract
from lib.visualize import visualize

conf = Config()
for i in conf.args:
    print(i,':', conf.args[i])

sum=0


AG_dataset = AG(mode="test", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                filter_small_box=False if conf.mode == 'predcls' else True)
#AG_dataset=object{object_classes,
#                  relationship_classes as rc,
#                  attention_relationship(rc[0:3]),
#                  sptial_relationship(rc[3:9]),
#                  contacting_relationship(rc[9:]),
#                  video_list,
#                  video_size,
#                  gt_annotations,
#                  non_gt_human_nums,
#                  non_heatmap_nums,
#                  non_person_video,
#                  one_frame_video,
#                  valid_nums}
dataloader = torch.utils.data.DataLoader(AG_dataset, shuffle=False, num_workers=0, collate_fn=cuda_collate_fn)
#shuffle:disrupt the data or not
#dataloader的属性是：img_tensor, im_info, gt_boxes, num_boxes, index
gpu_device = torch.device('cuda:0')
object_detector = detector(train=False, object_classes=AG_dataset.object_classes, use_SUPPLY=True, mode=conf.mode).to(device=gpu_device)
#['__background__', 'person', 'bag', 'bed', 'blanket', 'book',
#               'box', 'broom', 'chair', 'closet/cabinet', 'clothes',
#               'cup/glass/bottle', 'dish', 'door', 'doorknob', 'doorway',
#               'floor', 'food', 'groceries', 'laptop', 'light', 'medicine',
#               'mirror', 'paper/notebook', 'phone/camera', 'picture', 'pillow',
#               'refrigerator', 'sandwich', 'shelf', 'shoe', 'sofa/couch',
#               'table', 'television', 'towel', 'vacuum', 'window']
object_detector.eval()

model = STTran(mode=conf.mode,
               attention_class_num=len(AG_dataset.attention_relationships),
               #attention_relationship(rc[0:3])
               #['looking_at', 'not_looking_at', 'unsure']=3
               spatial_class_num=len(AG_dataset.spatial_relationships),
               #['above', 'beneath', 'in_front_of', 'behind', 'on_the_side_of', 'in']
               contact_class_num=len(AG_dataset.contacting_relationships),
               #['carrying', 'covered_by', 'drinking_from', 'eating', 'have_it_on_the_back', 'holding',
               #                        'leaning_on', 'lying_on', 'not_contacting', 'other_relationship',
               #                        'sitting_on', 'standing_on', 'touching', 'twisting', 'wearing',
               #                        'wiping', 'writing_on']
               obj_classes=AG_dataset.object_classes,
#['__background__', 'person', 'bag', 'bed', 'blanket', 'book',
#               'box', 'broom', 'chair', 'closet/cabinet', 'clothes',
#               'cup/glass/bottle', 'dish', 'door', 'doorknob', 'doorway',
#               'floor', 'food', 'groceries', 'laptop', 'light', 'medicine',
#               'mirror', 'paper/notebook', 'phone/camera', 'picture', 'pillow',
#               'refrigerator', 'sandwich', 'shelf', 'shoe', 'sofa/couch',
#               'table', 'television', 'towel', 'vacuum', 'window']
               enc_layer_num=conf.enc_layer,#1
               dec_layer_num=conf.dec_layer).to(device=gpu_device)#3

model.eval()

ckpt = torch.load(conf.model_path, map_location=gpu_device)
model.load_state_dict(ckpt['state_dict'], strict=False)
print('*'*50)
print('CKPT {} is loaded'.format(conf.model_path))
#

evaluator1 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset.object_classes,
    AG_all_predicates=AG_dataset.relationship_classes,
    AG_attention_predicates=AG_dataset.attention_relationships,
    AG_spatial_predicates=AG_dataset.spatial_relationships,
    AG_contacting_predicates=AG_dataset.contacting_relationships,
    iou_threshold=0.5,
    constraint='with')
evaluator2 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset.object_classes,
    AG_all_predicates=AG_dataset.relationship_classes,
    AG_attention_predicates=AG_dataset.attention_relationships,
    AG_spatial_predicates=AG_dataset.spatial_relationships,
    AG_contacting_predicates=AG_dataset.contacting_relationships,
    iou_threshold=0.5,
    constraint='semi', semithreshold=0.9)
evaluator3 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset.object_classes,
    AG_all_predicates=AG_dataset.relationship_classes,
    AG_attention_predicates=AG_dataset.attention_relationships,
    AG_spatial_predicates=AG_dataset.spatial_relationships,
    AG_contacting_predicates=AG_dataset.contacting_relationships,
    iou_threshold=0.5,
    constraint='no')
iter=1
with torch.no_grad():
    for b, data in enumerate(dataloader):
        #b=0,1,2...
        #data=img_tensor, im_info, gt_boxes, num_boxes, index
        #第一个文件夹就是00607.mp4
        #一个文件夹一个文件夹的输入
        iter=iter+1
        sum=sum+1
        im_data = copy.deepcopy(data[0].cuda(0))#38张
        im_info = copy.deepcopy(data[1].cuda(0))
        gt_boxes = copy.deepcopy(data[2].cuda(0))
        num_boxes = copy.deepcopy(data[3].cuda(0))
        gt_annotation = AG_dataset.gt_annotations[data[4]]#找到对应的文件夹的gt_annotations
        openpose_index = copy.deepcopy(data[5])
        img_h_l=copy.deepcopy(data[6])
        openpose_index_len=len(openpose_index)
        openposed_value_scale=im_info[0][2]
        frame_path=copy.deepcopy(data[8])



        a=time.time()
        entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
        b=time.time()
        pose= extract(False, openpose_index, openpose_index_len, openposed_value_scale,None,None)
        c=time.time()


        #print(entry['boxes'].shape)  torch.Size([322, 5])
        #predcls-->entry={box,labels,scores,im_idx,pair_idx,human_idx,feature,union_feat,union_box,
        #                                          spatial_masks,attention_gt,spatial_gt,contacting_gt}
        #sgdet-->entry={boxs,scores,distribution,pred_labels,features,fmaps,im_info}与relationship无关
        pred = model(False,entry,None)
        d=time.time()
        pred_bbox=pred['boxes'].cpu().clone().numpy()
        #pred['boxes'][:, 1:].cpu().clone().numpy()
        #pred={boxes,scores,distribution,pred_labels,featuresfmaps,im_info,pred_scores,pair_idx,im_idx,
        #                        human_idx,union_feat,union_box,spatial_masks,attention_distribution,
        #                        spatial_distribution,contacting_distribution}
        #1
        # print(len(gt_annotation))
        # print(len(dict(pred)))
        # print("hasfnkaw")
        # exit()
        rel_pred1,len_pics1=evaluator1.evaluate_scene_graph(gt_annotation, dict(pred))
        #2
        rel_pred2,len_pics2=evaluator2.evaluate_scene_graph(gt_annotation, dict(pred))
        #3
        rel_pred3,len_pics3=evaluator3.evaluate_scene_graph(gt_annotation, dict(pred))

        visualize(rel_pred1,len_pics1+1,frame_path,openposed_value_scale,pred_bbox)
        exit()
        print('iter {},fr consuming:{},pose consuming:{},sttran consuming:{}'.format(iter,b-a,c-b,d-c))
        print('rel_pred1:',rel_pred1)
        print('len:',len_pics1)

print("all iter:{}. now going to print results".format(sum))
print('-------------------------with constraint-------------------------------')
evaluator1.print_stats()
print('-------------------------semi constraint-------------------------------')
evaluator2.print_stats()
print('-------------------------no constraint-------------------------------')
evaluator3.print_stats()



'''all iter:1750. now going to print results
Predcls:
-------------------------with constraint-------------------------------
======================predcls============================
R@10: 0.685553
R@20: 0.717774
R@50: 0.717960
R@100: 0.717960
-------------------------semi constraint-------------------------------
======================predcls============================
R@10: 0.732235
R@20: 0.831277
R@50: 0.840061
R@100: 0.840061
-------------------------no constraint-------------------------------
======================predcls============================
R@10: 0.778570
R@20: 0.942340
R@50: 0.990808
R@100: 0.998785
'''
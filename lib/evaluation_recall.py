import torch
import torch.nn as nn
import numpy as np
from functools import reduce
from lib.ults.pytorch_misc import intersect_2d, argsort_desc
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps

class BasicSceneGraphEvaluator:
    def __init__(self, mode, AG_object_classes, AG_all_predicates, AG_attention_predicates, AG_spatial_predicates, AG_contacting_predicates,
                 iou_threshold=0.5, constraint=False, semithreshold=None):
        self.result_dict = {}
        self.mode = mode#SGDet
        self.result_dict[self.mode + '_recall'] = {10: [], 20: [], 50: [], 100: []}
        self.constraint = constraint#with/semi/no.   semi constraint if True
        self.iou_threshold = iou_threshold#0.5
        self.AG_object_classes = AG_object_classes#see above
        self.AG_all_predicates = AG_all_predicates#see before
        #['looking_at', 'not_looking_at', 'unsure', 'above', 'beneath', 'in_front_of', 'behind',
        #               'on_the_side_of', 'in', 'carrying', 'covered_by', 'drinking_from', 'eating',
        #               'have_it_on_the_back', 'holding', 'leaning_on', 'lying_on', 'not_contacting',
        #               'other_relationship', 'sitting_on', 'standing_on', 'touching', 'twisting',
        #               'wearing', 'wiping', 'writing_on']
        self.AG_attention_predicates = AG_attention_predicates#see before
        self.AG_spatial_predicates = AG_spatial_predicates#see before
        self.AG_contacting_predicates = AG_contacting_predicates#see before
        self.semithreshold = semithreshold#0.5

    def reset_result(self):
        self.result_dict[self.mode + '_recall'] = {10: [], 20: [], 50: [], 100: []}

    def print_stats(self,epoch,eval_path):
        print('======================' + self.mode + '============================')
        with open(eval_path, 'a', encoding='utf-8') as f:
            for k, v in self.result_dict[self.mode + '_recall'].items():
                print('R@%i: %f' % (k, np.mean(v)))
                f.write(str((k, np.mean(v))))
            f.close()

    # gt_annotations里是整个文件夹里所有图片的{person_box,object_box}
    # gt_annotations:[...,[...,[{person_box},{class,bbox,attention_relationship,...}],...],...]
    #                    -----------------gt_annotation_video----------------------------
    #                         ------------gt_annotation_frame-----------------------
    # [[[{'person_bbox': array([[140.7502, 150.3986, 229.9369, 328.0558]], dtype=float32)},
    # {'class': 32, 'bbox': array([208.2368, 229.0974, 270.    , 333.3916]),
    #           'attention_relationship': tensor([0]),
    #           'spatial_relationship': tensor([2]),
    #           'contacting_relationship': tensor([8]),
    #           'metadata': {'tag': '00607.mp4/table/000016', 'set': 'test'},
    #           'visible': True}],
    # [{'person_bbox': array([[158.8245, 131.2105, 255.9878, 354.1025]], dtype=float32)},
    #           {'class': 32, 'bbox': array([208.4464, 239.7143, 269.8031, 348.7546]),
    #           'attention_relationship': tensor([2]),
    #           'spatial_relationship': tensor([2]),
    #           'contacting_relationship': tensor([8]),
    #           'metadata': {'tag': '00607.mp4/table/000047', 'set': 'test'},
    #           'visible': True}],
    # [{'person_bbox': array([[160.8105, 213.7097, 251.6256, 356.7066]], dtype=float32)},
    #            {'class': 32, 'bbox': array([215.9343, 217.8608, 270.    , 350.1521]),
    #            'attention_relationship': tensor([1]),
    #            'spatial_relationship': tensor([0]),
    #            'contacting_relationship': tensor([12]),
    #            'metadata': {'tag': '00607.mp4/table/000077', 'set': 'test'},
    #            'visible': True}]]]
    def evaluate_scene_graph(self, gt, pred):
        '''collect the groundtruth and prediction'''
        #pred={boxes,scores,distribution,pred_labels,featuresfmaps,im_info,pred_scores,pair_idx,im_idx,
        #                        human_idx,union_feat,union_box,spatial_masks,attention_distribution,
        #                        spatial_distribution,contacting_distribution}
        pred['attention_distribution'] = nn.functional.softmax(pred['attention_distribution'], dim=1)
        rel_preds=[]
        for idx, frame_gt in enumerate(gt):
            # generate the ground truth
            len_pic=idx
            gt_boxes = np.zeros([len(frame_gt), 4]) #now there is no person box! we assume that person box index == 0
            gt_classes = np.zeros(len(frame_gt))
            gt_relations = []
            human_idx = 0
            gt_classes[human_idx] = 1
            gt_boxes[human_idx] = frame_gt[0]['person_bbox']
            for m, n in enumerate(frame_gt[1:]):
                # each pair
                gt_boxes[m+1,:] = n['bbox']
                gt_classes[m+1] = n['class']
                gt_relations.append([human_idx, m+1, self.AG_all_predicates.index(self.AG_attention_predicates[n['attention_relationship']])]) # for attention triplet <human-object-predicate>_
                #spatial and contacting relationship could be multiple
                for spatial in n['spatial_relationship'].numpy().tolist():
                    gt_relations.append([m+1, human_idx, self.AG_all_predicates.index(self.AG_spatial_predicates[spatial])]) # for spatial triplet <object-human-predicate>
                for contact in n['contacting_relationship'].numpy().tolist():
                    gt_relations.append([human_idx, m+1, self.AG_all_predicates.index(self.AG_contacting_predicates[contact])])  # for contact triplet <human-object-predicate>

            gt_entry = {
                'gt_classes': gt_classes,
                'gt_relations': np.array(gt_relations),
                'gt_boxes': gt_boxes,
            }

            # first part for attention and contact, second for spatial

            rels_i = np.concatenate((pred['pair_idx'][pred['im_idx'] == idx].cpu().clone().numpy(),             #attention
                                     pred['pair_idx'][pred['im_idx'] == idx].cpu().clone().numpy()[:,::-1],     #spatial
                                     pred['pair_idx'][pred['im_idx'] == idx].cpu().clone().numpy()), axis=0)    #contacting
            #print(pred['pair_idx'])

            # tensor([[0, 1],
            #         [0, 2],
            #         [0, 3],
            #         [0, 4],
            #         [0, 5],
            #         [0, 6],
            #         [0, 7],
            #         [0, 8],
            #         [0, 9],
            #         [10, 11],
            #         [10, 12],
            #         [10, 13],
            #         [10, 14],
            #         [10, 15],
            #         [10, 16],
            #         [17, 18],
            #         [17, 19],
            #         [17, 20],
            #         [17, 21],
            #         [17, 22],
            #         [23, 24],
            #         [23, 25],
            #         [23, 26],
            #         [23, 27],
            #         [23, 28],
            #         [23, 29],
            #         [23, 30],
            #         [23, 31],
            #         [32, 33],
            #         [32, 34],
            #         [32, 35],
            #         [32, 36],
            #         [32, 37],
            #         [32, 38],
            #         [32, 39],
            #         [32, 40],
            #         [32, 41],
            #         [42, 43],
            #         [42, 44],
            #         [42, 45],
            #         [42, 46],
            #         [47, 48],
            #         [47, 49],
            #         [47, 50],
            #         [47, 51],
            #         [47, 52],
            #         [47, 53],
            #         [47, 54],
            #         [47, 55],
            #         [47, 56],
            #         [47, 57],
            #         [58, 59],
            #         [58, 60],
            #         [58, 61],
            #         [58, 62],
            #         [58, 63],
            #         [58, 64],
            #         [58, 65],
            #         [66, 67],
            #         [66, 68],
            #         [66, 69],
            #         [66, 70],
            #         [66, 71],
            #         [66, 72],
            #         [66, 73],
            #         [66, 74],
            #         [75, 76],
            #         [75, 77],
            #         [75, 78],
            #         [75, 79],
            #         [75, 80],
            #         [81, 82],
            #         [81, 83],
            #         [81, 84],
            #         [81, 85],
            #         [81, 86],
            #         [81, 87],
            #         [81, 88],
            #         [81, 89],
            #         [81, 90],
            #         [81, 91],
            #         [81, 92],
            #         [81, 93],
            #         [94, 95],
            #         [94, 96],
            #         [94, 97],
            #         [94, 98],
            #         [94, 99],
            #         [94, 100],
            #         [94, 101],
            #         [102, 103],
            #         [102, 104],
            #         [102, 105],
            #         [102, 106],
            #         [102, 107],
            #         [102, 108],
            #         [109, 110],
            #         [109, 111],
            #         [109, 112],
            #         [109, 113],
            #         [109, 114],
            #         [109, 115],
            #         [116, 117],
            #         [116, 118],
            #         [116, 119],
            #         [120, 121],
            #         [120, 122],
            #         [120, 123],
            #         [120, 124],
            #         [120, 125],
            #         [120, 126],
            #         [127, 128],
            #         [127, 129],
            #         [127, 130],
            #         [127, 131],
            #         [127, 132],
            #         [127, 133],
            #         [134, 135],
            #         [134, 136],
            #         [134, 137],
            #         [138, 139],
            #         [138, 140],
            #         [138, 141],
            #         [138, 142],
            #         [138, 143],
            #         [138, 144],
            #         [138, 145],
            #         [138, 146],
            #         [138, 147],
            #         [148, 149],
            #         [148, 150],
            #         [148, 151],
            #         [148, 152],
            #         [148, 153],
            #         [148, 154],
            #         [148, 155],
            #         [148, 156],
            #         [148, 157],
            #         [148, 158],
            #         [159, 160],
            #         [159, 161],
            #         [159, 162],
            #         [159, 163],
            #         [159, 164],
            #         [159, 165],
            #         [159, 166],
            #         [159, 167],
            #         [168, 169],
            #         [168, 170],
            #         [168, 171],
            #         [168, 172],
            #         [168, 173],
            #         [168, 174],
            #         [168, 175],
            #         [176, 177],
            #         [176, 178],
            #         [176, 179],
            #         [176, 180],
            #         [181, 182],
            #         [181, 183],
            #         [181, 184],
            #         [181, 185],
            #         [181, 186],
            #         [181, 187],
            #         [181, 188],
            #         [189, 190],
            #         [189, 191],
            #         [189, 192],
            #         [189, 193],
            #         [189, 194],
            #         [189, 195],
            #         [196, 197],
            #         [196, 198],
            #         [196, 199],
            #         [196, 200],
            #         [196, 201],
            #         [202, 203],
            #         [202, 204],
            #         [202, 205],
            #         [202, 206],
            #         [202, 207],
            #         [208, 209],
            #         [208, 210],
            #         [208, 211],
            #         [208, 212],
            #         [208, 213],
            #         [208, 214],
            #         [208, 215],
            #         [216, 217],
            #         [216, 218],
            #         [216, 219],
            #         [216, 220],
            #         [221, 222],
            #         [221, 223],
            #         [221, 224],
            #         [221, 225],
            #         [221, 226],
            #         [227, 228],
            #         [227, 229],
            #         [227, 230],
            #         [227, 231],
            #         [227, 232],
            #         [233, 234],
            #         [233, 235],
            #         [233, 236],
            #         [233, 237],
            #         [233, 238],
            #         [233, 239],
            #         [233, 240],
            #         [233, 241],
            #         [242, 243],
            #         [242, 244],
            #         [242, 245],
            #         [242, 246],
            #         [242, 247],
            #         [242, 248],
            #         [242, 249],
            #         [250, 251],
            #         [250, 252],
            #         [250, 253],
            #         [254, 255],
            #         [254, 256],
            #         [254, 257],
            #         [258, 259],
            #         [258, 260],
            #         [258, 261],
            #         [258, 262],
            #         [258, 263],
            #         [264, 265],
            #         [264, 266],
            #         [264, 267],
            #         [264, 268],
            #         [269, 270],
            #         [269, 271],
            #         [269, 272],
            #         [269, 273],
            #         [269, 274],
            #         [269, 275],
            #         [269, 276],
            #         [269, 277]], device='cuda:0')

            #print(rels_i)

           # [[0 1]
           #  [0 2]
           #  [0 3]
           #  [0 4]
           #  [0 5]
           #  [0 6]
           #  [0 7]
           #  [0 8]
           #  [0 9]
           #  [1 0]
           #  [2 0]
           #  [3 0]
           #  [4 0]
           #  [5 0]
           #  [6 0]
           #  [7 0]
           #  [8 0]
           #  [9 0]
           #  [0 1]
           #  [0 2]
           #  [0 3]
           #  [0 4]
           #  [0 5]
           #  [0 6]
           #  [0 7]
           #  [0 8]
           #  [0 9]]

            #print(pred['im_idx'] == idx)

            # tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            #        device='cuda:0', dtype=torch.uint8)

            pred_scores_1 = np.concatenate((pred['attention_distribution'][pred['im_idx'] == idx].cpu().numpy(),
                                            np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['spatial_distribution'].shape[1]]),
                                            np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['contacting_distribution'].shape[1]])), axis=1)
            pred_scores_2 = np.concatenate((np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['attention_distribution'].shape[1]]),
                                            pred['spatial_distribution'][pred['im_idx'] == idx].cpu().numpy(),
                                            np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['contacting_distribution'].shape[1]])), axis=1)
            pred_scores_3 = np.concatenate((np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['attention_distribution'].shape[1]]),
                                            np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['spatial_distribution'].shape[1]]),
                                            pred['contacting_distribution'][pred['im_idx'] == idx].cpu().numpy()), axis=1)

            if self.mode == 'predcls':

                pred_entry = {
                    'pred_boxes': pred['boxes'][:,1:].cpu().clone().numpy(),
                    'pred_classes': pred['labels'].cpu().clone().numpy(),
                    'pred_rel_inds': rels_i,
                    'obj_scores': pred['scores'].cpu().clone().numpy(),
                    'rel_scores': np.concatenate((pred_scores_1, pred_scores_2, pred_scores_3), axis=0)
                }
            else:
                pred_entry = {
                    'pred_boxes': pred['boxes'][:, 1:].cpu().clone().numpy(),
                    'pred_classes': pred['pred_labels'].cpu().clone().numpy(),
                    'pred_rel_inds': rels_i,#[attention,spatial,contacting]
                    'obj_scores': pred['pred_scores'].cpu().clone().numpy(),
                    'rel_scores': np.concatenate((pred_scores_1, pred_scores_2, pred_scores_3), axis=0)
                }

            _,rel_pred,__=evaluate_from_dict(gt_entry, pred_entry, self.mode, self.result_dict,
                               iou_thresh=self.iou_threshold, method=self.constraint, threshold=self.semithreshold)
            rel_preds.append(rel_pred)

        return rel_preds,len_pic

def evaluate_from_dict(gt_entry, pred_entry, mode, result_dict, method=None, threshold = 0.9, **kwargs):
    """
    Shortcut to doing evaluate_recall from dict
    :param gt_entry: Dictionary containing gt_relations, gt_boxes, gt_classes
    :param pred_entry: Dictionary containing pred_rels, pred_boxes (if detection), pred_classes
    :param result_dict:
    :param kwargs:
    :return:
    """
    rel=[]
    gt_rels = gt_entry['gt_relations']#真正的relationship
    gt_boxes = gt_entry['gt_boxes'].astype(float)
    gt_classes = gt_entry['gt_classes']

    # 'pred_rel_inds': array([[0, 1],
    #                         [0, 2],
    #                         [0, 3],
    #                         [0, 4],
    #                         [0, 5],
    #                         [0, 6],
    #                         [0, 7],
    #                         [0, 8],
    #                         [0, 9],
    #                         [1, 0],
    #                         [2, 0],
    #                         [3, 0],
    #                         [4, 0],
    #                         [5, 0],
    #                         [6, 0],
    #                         [7, 0],
    #                         [8, 0],
    #                         [9, 0],
    #                         [0, 1],
    #                         [0, 2],
    #                         [0, 3],
    #                         [0, 4],
    #                         [0, 5],
    #                         [0, 6],
    #                         [0, 7],
    #                         [0, 8],
    #                         [0, 9]])
    pred_rel_inds = pred_entry['pred_rel_inds']#[attention,spatial,contacting]
    rel_scores = pred_entry['rel_scores']#预测的关系score

    pred_boxes = pred_entry['pred_boxes'].astype(float)
    pred_classes = pred_entry['pred_classes']
    obj_scores = pred_entry['obj_scores']

    if method == 'semi':
        pred_rels = []
        predicate_scores = []
        for i, j in enumerate(pred_rel_inds):
            if rel_scores[i,0]+rel_scores[i,1] > 0:
                # this is the attention distribution
                pred_rels.append(np.append(j,rel_scores[i].argmax()))
                predicate_scores.append(rel_scores[i].max())
            elif rel_scores[i,3]+rel_scores[i,4] > 0:
                # this is the spatial distribution
                for k in np.where(rel_scores[i]>threshold)[0]:
                    pred_rels.append(np.append(j, k))
                    predicate_scores.append(rel_scores[i,k])
            elif rel_scores[i,9]+rel_scores[i,10] > 0:
                # this is the contact distribution
                for k in np.where(rel_scores[i]>threshold)[0]:
                    pred_rels.append(np.append(j, k))
                    predicate_scores.append(rel_scores[i,k])

        pred_rels = np.array(pred_rels)
        predicate_scores = np.array(predicate_scores)
    elif method == 'no':
        obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
        overall_scores = obj_scores_per_rel[:, None] * rel_scores
        score_inds = argsort_desc(overall_scores)[:100]
        pred_rels = np.column_stack((pred_rel_inds[score_inds[:, 0]], score_inds[:, 1]))
        predicate_scores = rel_scores[score_inds[:, 0], score_inds[:, 1]]

    else:
        pred_rels = np.column_stack((pred_rel_inds, rel_scores.argmax(1))) #1+  dont add 1 because no dummy 'no relations'分数索引和关系合并
        predicate_scores = rel_scores.max(1)

    # print("pred_rel_inds",pred_rel_inds)
    # print("pred_rels:",pred_rels)
    # print("pred_boxes",pred_boxes)
    # print("pred_classes",pred_classes)
    #
    # print("len pred_rels:",len(pred_rels))
    # print("len pred_boxes",len(pred_boxes))
    # print("len pred_classes",len(pred_classes))
    # exit()
    pred_to_gt, pred_5ples, rel_scores = evaluate_recall(
        #pred_to_gt: Matching from predicate to GT
        #pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
                gt_rels, gt_boxes, gt_classes,
                pred_rels, pred_boxes, pred_classes,
                predicate_scores, obj_scores, phrdet= mode=='phrdet',
                **kwargs)

    #此处为输出
    #print(pred_to_gt)
    #[[], [], [], [], [2], [1], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]

    # print(pred_5ples)
    # [[0  1  1  7 17]
    #  [0  2  7  1  5]
    # [0 3 16 1 4]
    # [0  4  1 16 20]
    # [0 5 1    32    17]
    # [0  6 32  1  5]
    # [0 7    1 32    1]
    # [0  8  1 29 17]
    # [0 9 29  1 7]
    # [1  0  1 29  1]]
    for k in result_dict[mode + '_recall']:

        match = reduce(np.union1d, pred_to_gt[:k])

        rec_i = float(len(match)) / float(gt_rels.shape[0])
        result_dict[mode + '_recall'][k].append(rec_i)
    return pred_to_gt, pred_5ples, rel_scores

###########################
def evaluate_recall(gt_rels, gt_boxes, gt_classes,
                    pred_rels, pred_boxes, pred_classes, rel_scores=None, cls_scores=None,
                    iou_thresh=0.5, phrdet=False):
    """
    Evaluates the recall
    :param gt_rels: [#gt_rel, 3] array of GT relations
    :param gt_boxes: [#gt_box, 4] array of GT boxes
    :param gt_classes: [#gt_box] array of GT classes
    :param pred_rels: [#pred_rel, 3] array of pred rels. Assumed these are in sorted order
                      and refer to IDs in pred classes / pred boxes
                      (id0, id1, rel)
    :param pred_boxes:  [#pred_box, 4] array of pred boxes
    :param pred_classes: [#pred_box] array of predicted classes for these boxes
    :return: pred_to_gt: Matching from predicate to GT
             pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
             rel_scores: [cls_0score, cls1_score, relscore]
                   """
    # print('aaa')
    # print(gt_rels)
    # print('bbb')
    # print(gt_boxes)
    # print('cccc')
    # print(gt_classes)
    # exit()
    if pred_rels.size == 0:
        return [[]], np.zeros((0,5)), np.zeros(0)

    num_gt_boxes = gt_boxes.shape[0]
    num_gt_relations = gt_rels.shape[0]
    assert num_gt_relations != 0

    gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels[:, 2],
                                                gt_rels[:, :2],
                                                gt_classes,
                                                gt_boxes)
    num_boxes = pred_boxes.shape[0]
    assert pred_rels[:,:2].max() < pred_classes.shape[0]

    # Exclude self rels
    # assert np.all(pred_rels[:,0] != pred_rels[:,ĺeftright])
    #assert np.all(pred_rels[:,2] > 0)

    pred_triplets, pred_triplet_boxes, relation_scores = \
        _triplet(pred_rels[:,2], pred_rels[:,:2], pred_classes, pred_boxes,
                 rel_scores, cls_scores)

    sorted_scores = relation_scores.prod(1)
    pred_triplets = pred_triplets[sorted_scores.argsort()[::-1],:]
    pred_triplet_boxes = pred_triplet_boxes[sorted_scores.argsort()[::-1],:]
    relation_scores = relation_scores[sorted_scores.argsort()[::-1],:]
    scores_overall = relation_scores.prod(1)

    if not np.all(scores_overall[1:] <= scores_overall[:-1] + 1e-5):
        print("Somehow the relations weren't sorted properly: \n{}".format(scores_overall))
        # raise ValueError("Somehow the relations werent sorted properly")

    # Compute recall. It's most efficient to match once and then do recall after
    pred_to_gt = _compute_pred_matches(
        gt_triplets,
        pred_triplets,
        gt_triplet_boxes,
        pred_triplet_boxes,
        iou_thresh,
        phrdet=phrdet,
    )

    # Contains some extra stuff for visualization. Not needed.
    pred_5ples = np.column_stack((
        pred_rels[:,:2],
        pred_triplets[:, [0, 2, 1]],
    ))

    return pred_to_gt, pred_5ples, relation_scores


def _triplet(predicates, relations, classes, boxes,
             predicate_scores=None, class_scores=None):
    """
    format predictions into triplets
    :param predicates: A 1d numpy array of num_boxes*(num_boxes-ĺeftright) predicates, corresponding to
                       each pair of possibilities
    :param relations: A (num_boxes*(num_boxes-ĺeftright), 2.0) array, where each row represents the boxes
                      in that relation
    :param classes: A (num_boxes) array of the classes for each thing.
    :param boxes: A (num_boxes,4) array of the bounding boxes for everything.
    :param predicate_scores: A (num_boxes*(num_boxes-ĺeftright)) array of the scores for each predicate
    :param class_scores: A (num_boxes) array of the likelihood for each object.
    :return: Triplets: (num_relations, 3) array of class, relation, class
             Triplet boxes: (num_relation, 8) array of boxes for the parts
             Triplet scores: num_relation array of the scores overall for the triplets
    """
    assert (predicates.shape[0] == relations.shape[0])

    sub_ob_classes = classes[relations[:, :2]]
    triplets = np.column_stack((sub_ob_classes[:, 0], predicates, sub_ob_classes[:, 1]))
    triplet_boxes = np.column_stack((boxes[relations[:, 0]], boxes[relations[:, 1]]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[relations[:, 0]],
            class_scores[relations[:, 1]],
            predicate_scores,
        ))

    return triplets, triplet_boxes, triplet_scores


def _compute_pred_matches(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thresh, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    :param gt_triplets:
    :param pred_triplets:
    :param gt_boxes:
    :param pred_boxes:
    :param iou_thresh:
    :return:
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:,:2], box_union.max(1)[:,2:]), 1)

            inds = bbox_overlaps(gt_box_union[None], box_union)[0] >= iou_thresh

        else:
            sub_iou = bbox_overlaps(gt_box[None,:4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None,4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt

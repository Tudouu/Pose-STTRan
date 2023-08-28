import numpy as np
from lib.fpn.box_utils import bbox_overlaps
import cv2


def assign_relations(prediction, gt_annotations, assign_IOU_threshold):
    '''
    :param prediction(list): results from FasterRCNN, each element is a dictionary including the predicted boxes,
                            labels, scores, base_feature(image), features(rois), im_info (w,h,scale)
    :param gt_annotations(list):  ground-truth, each element is a list including person info(always element 0) and objects
    :param assign_IOU_threshold: hyperparameter for SGDET, 0.5
    :return: DETECTOR_FOUND_IDX
             GT_RELATIONS
             SUPPLY_RELATIONS
    '''
    FINAL_BBOXES = prediction['FINAL_BBOXES']#[box_num, 5],第一列是图片的编号从0开始
    FINAL_LABELS = prediction['FINAL_LABELS']
    DETECTOR_FOUND_IDX = []
    GT_RELATIONS = []
    SUPPLY_RELATIONS = []
    #print(FINAL_LABELS.shape)  # [label_num]
    #print(FINAL_BBOXES.shape)  #[box_num, 5]

    assigned_labels = np.zeros(FINAL_LABELS.shape[0])#标签的数量
    # gt_annotations里是整个文件夹里所有图片的{person_box,object_box}
    # gt_annotations:[...,[...,[{person_box},{class,bbox,attention_relationship,...}],...],...]
    #                    --------------gt_annotation_video(一张图里所有框)----------------
    #                         -----gt_annotation_frame(一张图里的一对框)------------
    # j=gt_annotation_video
    # gt_annotation_video=gt_annotation_video.append(gt_annotation_frame)
    # gt_annotation_frame=[{'person_bbox': person_bbox['001YG.mp4/000089.png']['bbox']},
    #                     {class,bbox,attention_relationship,...}]
    # gt_annotation=[...,[{'person_bbox': array([[131.104, 67.112, 254.437, 264.468]], dtype=float32)},
    # {'class': 15, 'bbox': array([0., 0., 163.611, 269.956]), 'attention_relationship': tensor([1]),
    # 'spatial_relationship': tensor([4]), 'contacting_relationship': tensor([12]),
    # 'metadata': {'tag': 'HAA4O.mp4/doorway/000216', 'set': 'train'}, 'visible': True},
    # {'class': 13, 'bbox': array([2., 2.5, 161.76, 267.5]), 'attention_relationship': tensor([1]),
    # 'spatial_relationship': tensor([4]), 'contacting_relationship': tensor([12]),
    # 'metadata': {'tag': 'HAA4O.mp4/door/000216', 'set': 'train'}, 'visible': True}],...]
    for i, j in enumerate(gt_annotations):

        gt_boxes = np.zeros([len(j), 4])#每张图片所有框坐标,len(j)是一张图片里框的数量
        gt_labels = np.zeros(len(j))#标签数量
        gt_boxes[0] = j[0]['person_bbox']#把人的坐标存进gt_box第一行
        gt_labels[0] = 1
        for m, n in enumerate(j[1:]):#j[1:]是这张图片的所有class
            #[{'class': 15, 'bbox': [0.0, 0.0, 163.611, 269.956], 'attention_relationship': [1], 'spatial_relationship': [4], 'contacting_relationship': [12], 'metadata': {'tag': 'HAA4O.mp4/doorway/000216', 'set': 'train'}, 'visible': True},
            # {'class': 13, 'bbox': [2.0, 2.5, 161.76, 267.5], 'attention_relationship': [1], 'spatial_relationship': [4], 'contacting_relationship': [12], 'metadata': {'tag': 'HAA4O.mp4/door/000216', 'set': 'train'}, 'visible': True}]
            #提取所有的物品框
            gt_boxes[m+1,:] = n['bbox']#把所有行都填满了，第一行是人坐标，剩下的是物体的坐标
            gt_labels[m+1] = n['class']#第一个是人标签，剩下是物体标签

        #gt_boxes[0]为人坐标,gt_boxes[1:]为物体坐标
        #gt_labels[0]为人标签1,gt_labels[1:]为物体标签
        pred_boxes = FINAL_BBOXES[FINAL_BBOXES[:,0] == i, 1:].detach().cpu().numpy()
        #print('pred_boxes',pred_boxes)
        #找到对应的pred,这个pred_boxes是i=0时候的框
        #labels = FINAL_LABELS[FINAL_BBOXES[:,0] == i].detach().cpu().numpy()
        #print('predbox',pred_boxes.shape) [这张图片的box_num,4]
        #print('gtbox',gt_boxes.shape) [这张图片的gtbox_num,4]
        IOUs = bbox_overlaps(pred_boxes, gt_boxes)
        #print('ious',IOUs)#几行几列就是第某个pred与第某个gt的交并比
        #计算交并比
        IOUs_bool = IOUs > assign_IOU_threshold
        #
        assigned_labels[(FINAL_BBOXES[:, 0].cpu().numpy() == i).nonzero()[0][np.max(IOUs, axis=1)> 0.5]] = gt_labels[np.argmax(IOUs, axis=1)][np.max(IOUs, axis=1)> 0.5]
        #把符合iou条件的box的label挑出来
        detector_found_idx = []
        gt_relations = []
        supply_relations = []
        candidates = []
        for m, n in enumerate(gt_annotations[i]):
            if m == 0:
                # 1 is the person index, np.where find out the pred_boxes and check which one corresponds to gt_label
                if sum(IOUs[:, m]>assign_IOU_threshold) > 0:#assign_IOU_threshold=0.5，这个sum>0是一定的
                    candidate = IOUs[:, m].argmax() #[labels[np.where(IOUs_bool[:, m])[0]] == 1] 交并比最大的那个人的框，这个值是行的索引值，也就是pred的索引值
                    detector_found_idx.append(candidate)#detector_found_idx的第一个值是人框的索引
                    gt_relations.append(n)#这里把[{'person_bbox': array([[ 39.114,   9.168, 162.43 , 265.481]], dtype=float32)}]塞进去
                    candidates.append(candidate)#index存进candidates
                else:
                    supply_relations.append(n) #no person box is found...i think it is rarely(impossible)
            else:
                if sum(IOUs[:, m]>assign_IOU_threshold) > 0:
                    candidate = IOUs[:, m].argmax()#取最大值
                    if candidate in candidates:
                        # one predbox is already assigned with one gtbox
                        for c in np.argsort(-IOUs[:, m]):#加上-变成从大到小排列
                            if c not in candidates:
                                candidate = c
                                break
                    detector_found_idx.append(candidate)
                    gt_relations.append(n)
                    #{'class': 12, 'bbox': array([132.418, 191.078, 183.075, 232.92 ]), 'attention_relationship': tensor([2]),
                    # 'spatial_relationship': tensor([2]), 'contacting_relationship': tensor([12]),
                    # 'metadata': {'tag': 'AVSN8.mp4/dish/000196', 'set': 'train'}, 'visible': True}
                    candidates.append(candidate)#最后装的都是每个物体的pred框
                    assigned_labels[(FINAL_BBOXES[:, 0].cpu().numpy() == i).nonzero()[0][candidate]] = n['class']
                else:
                    #no overlapped box 这一列的pred没有和任何gt相交，也就是说没有pred到类gt这个框
                    supply_relations.append(n)
                    #把n=[{'class': 31, 'bbox': array([209., 173., 433., 285.]), 'attention_relationship': tensor([1]),
                    # 'spatial_relationship': tensor([1, 3]), 'contacting_relationship': tensor([10,  6]),
                    # 'metadata': {'tag': 'RJFT8.mp4/sofa_couch/000584', 'set': 'train'}, 'visible': True}]塞进去
        DETECTOR_FOUND_IDX.append(detector_found_idx)
        GT_RELATIONS.append(gt_relations)#其实就是把这张图片的里字典append在了一起
        #print('gt',GT_RELATIONS)
        #[[{'person_bbox': array([[180.666, 67.97, 277.425, 247.034]], dtype=float32)},
         # {'class': 17, 'bbox': array([186.193, 95.721, 204.877, 112.037]), 'attention_relationship': tensor([0]),
         #                          'spatial_relationship': tensor([2]), 'contacting_relationship': tensor([5, 3]),
          #                         'metadata': {'tag': 'J7TT5.mp4/food/000036', 'set': 'train'}, 'visible': True},
          #{'class': 16, 'bbox': array([115., 80.5, 322., 270.]), 'attention_relationship': tensor([1]),
          #                         'spatial_relationship': tensor([3, 1]), 'contacting_relationship': tensor([7]),
          #                         'metadata': {'tag': 'J7TT5.mp4/floor/000036', 'set': 'train'}, 'visible': True}]]
        SUPPLY_RELATIONS.append(supply_relations)

    #print('254',SUPPLY_RELATIONS)

    return DETECTOR_FOUND_IDX, GT_RELATIONS, SUPPLY_RELATIONS, assigned_labels



def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= np.array([[[102.9801, 115.9465, 122.7717]]])

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in [600]:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > 1000:
      im_scale = float(1000) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob


def enumerate_by_image(im_inds):
    im_inds_np = im_inds.cpu().numpy()
    initial_ind = int(im_inds_np[0])
    s = 0
    for i, val in enumerate(im_inds_np):
        if val != initial_ind:
            yield initial_ind, s, i
            initial_ind = int(val)
            s = i
    yield initial_ind, s, len(im_inds_np)

def transpose_packed_sequence_inds(lengths):
    """
    Goes from a TxB packed sequence to a BxT or vice versa. Assumes that nothing is a variable
    :param ps: PackedSequence
    :return:
    """

    new_inds = []
    new_lens = []
    cum_add = np.cumsum([0] + lengths)
    max_len = lengths[0]
    length_pointer = len(lengths) - 1
    for i in range(max_len):
        while length_pointer > 0 and lengths[length_pointer] <= i:
            length_pointer -= 1
        new_inds.append(cum_add[:(length_pointer+1)].copy())
        cum_add[:(length_pointer+1)] += 1
        new_lens.append(length_pointer+1)
    new_inds = np.concatenate(new_inds, 0)
    return new_inds, new_lens

def pad_sequence(frame_idx):

    lengths = []
    for i, s, e in enumerate_by_image(frame_idx):  # i img_index s:start_idx e:end_idx
        lengths.append(e - s)
    lengths = sorted(lengths, reverse=True)

    _, ls_transposed = transpose_packed_sequence_inds(lengths)  # move it to TxB form
    return ls_transposed

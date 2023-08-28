import pickle
import os
import torch
import numpy as np

root_path='/media/wow/disk2/AG/dataset'
with open(root_path + '/annotations/person_bbox.pkl', 'rb') as f:
    person_bbox= pickle.load(f)
f.close()
with open(root_path + '/annotations/object_bbox_and_relationship.pkl', 'rb') as k:
    object_bbox = pickle.load(k)
k.close()
"""for keys,values in person_bbox.items():
    print(keys+":"+str(values))
    exit(0)"""
"""num=0
for j in person_bbox.keys():
    print(j)
    print(object_bbox[j])
    exit(0)
for j in person_bbox.keys():
      if num==25:
          exit(0)
      else:
          num=num+1
          print(j)
          for h in object_bbox[j]:
              print(h)"""
"""video_dict={}
for i in person_bbox.keys():
    if object_bbox[i][0]['metadata']['set'] == 'test':  # train or testing?
        frame_valid = False
        for j in object_bbox[i]:  # the frame is valid if there is visible bbox
            if j['visible']:#those that are visible 
                frame_valid = True
        if frame_valid:
            video_name, frame_num = i.split('/')
            if video_name in video_dict.keys():
                video_dict[video_name].append(i)
            else:
                video_dict[video_name] = [i]
#for key,value in video_dict.items():
#   print(key+":"+str(value))
filter_nonperson_box_frame=True
non_gt_human_nums=0
valid_nums=0
for i in video_dict.keys():
    video = []
    gt_annotation_video = []
    for j in video_dict[i]:
        if filter_nonperson_box_frame:
            if person_bbox[j]['bbox'].shape[0] == 0:
                non_gt_human_nums += 1
                continue
            else:
                video.append(j)
                valid_nums += 1
"""
"""object_classes = ['__background__']
with open(os.path.join(root_path, 'annotations/object_classes.txt'), 'r') as f:
    for line in f.readlines():
        line = line.strip('\n')
        object_classes.append(line)
f.close()
object_classes[9] = 'closet/cabinet'
object_classes[11] = 'cup/glass/bottle'
object_classes[23] = 'paper/notebook'
object_classes[24] = 'phone/camera'
object_classes[31] = 'sofa/couch'"""

relationship_classes = []
with open(os.path.join(root_path, 'annotations/relationship_classes.txt'), 'r') as f:
    for line in f.readlines():
        line = line.strip('\n')
        relationship_classes.append(line)
f.close()
relationship_classes[0] = 'looking_at'
relationship_classes[1] = 'not_looking_at'
relationship_classes[5] = 'in_front_of'
relationship_classes[7] = 'on_the_side_of'
relationship_classes[10] = 'covered_by'
relationship_classes[11] = 'drinking_from'
relationship_classes[13] = 'have_it_on_the_back'
relationship_classes[15] = 'leaning_on'
relationship_classes[16] = 'lying_on'
relationship_classes[17] = 'not_contacting'
relationship_classes[18] = 'other_relationship'
relationship_classes[19] = 'sitting_on'
relationship_classes[20] = 'standing_on'
relationship_classes[25] = 'writing_on'

attention_relationships = relationship_classes[0:3]
spatial_relationships = relationship_classes[3:9]
contacting_relationships = relationship_classes[9:]

for l in object_bbox["001YG.mp4/000089.png"]:
    print(l)
    z=torch.tensor([attention_relationships.index(r) for r in l['attention_relationship']], dtype=torch.long)
    print(z)
"""for k in object_bbox["001YG.mp4/000089.png"]:
    if k['visible']:
        assert k['bbox'] != None, 'warning! The object is visible without bbox'
        k['class'] = object_classes.index(k['class'])
        k['bbox'] = np.array([k['bbox'][0], k['bbox'][1], k['bbox'][0] + k['bbox'][2], k['bbox'][1] + k['bbox'][3]])  # from xywh to xyxy
        k['attention_relationship'] = torch.tensor(
            [self.attention_relationships.index(r) for r in k['attention_relationship']], dtype=torch.long)
        k['spatial_relationship'] = torch.tensor(
            [self.spatial_relationships.index(r) for r in k['spatial_relationship']], dtype=torch.long)
        k['contacting_relationship'] = torch.tensor(
            [self.contacting_relationships.index(r) for r in k['contacting_relationship']], dtype=torch.long)
        gt_annotation_frame.append(k)"""

"""a = FINAL_BBOXES[0]
path = '/media/wow/disk2/AG/dataset/frames/00607.mp4/000016.png'
img = cv2.imread(path)
cv2.rectangle(img, (int(a[1]), int(a[2])), (int(a[3]), int(a[4])), (0, 255, 0), 3)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(a)
print(PRED_LABELS)
exit()
for keys,values in entry.items():
       print(keys)
exit()
"""
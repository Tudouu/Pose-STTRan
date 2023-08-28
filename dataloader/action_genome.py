import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import random
import imageio
import matplotlib.pyplot as plt
#from scipy.misc import imread
import numpy as np
import pickle
import os
from fasterRCNN.lib.model.utils.blob import prep_im_for_blob, im_list_to_blob


class AG(Dataset):

    def __init__(self, mode, datasize, data_path=None, filter_nonperson_box_frame=True, filter_small_box=False):

        root_path = data_path
        self.frames_path = os.path.join(root_path, 'frames/')

        # collect the object classes
        self.object_classes = ['__background__']
        with open(os.path.join(root_path, 'annotations/object_classes.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.object_classes.append(line)
        f.close()
        self.object_classes[9] = 'closet/cabinet'
        self.object_classes[11] = 'cup/glass/bottle'
        self.object_classes[23] = 'paper/notebook'
        self.object_classes[24] = 'phone/camera'
        self.object_classes[31] = 'sofa/couch'

        #__background__,person,bag,bed,blanket,book,box,broom,chair,closet / cabinet,
        #clothes,cup / glass / bottle,dish,door,doorknob,doorway,floor,food,groceries,
        #laptop,light,medicine,mirror,paper/notebook,phone/camera,picture,pillow,\
        #refrigerator,sandwich,shelf,shoe,sofa / couch,table,television,towel,vacuum,window

        # collect relationship classes
        self.relationship_classes = []
        with open(os.path.join(root_path, 'annotations/relationship_classes.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.relationship_classes.append(line)
        f.close()
        self.relationship_classes[0] = 'looking_at'
        self.relationship_classes[1] = 'not_looking_at'
        self.relationship_classes[5] = 'in_front_of'
        self.relationship_classes[7] = 'on_the_side_of'
        self.relationship_classes[10] = 'covered_by'
        self.relationship_classes[11] = 'drinking_from'
        self.relationship_classes[13] = 'have_it_on_the_back'
        self.relationship_classes[15] = 'leaning_on'
        self.relationship_classes[16] = 'lying_on'
        self.relationship_classes[17] = 'not_contacting'
        self.relationship_classes[18] = 'other_relationship'
        self.relationship_classes[19] = 'sitting_on'
        self.relationship_classes[20] = 'standing_on'
        self.relationship_classes[25] = 'writing_on'

        self.attention_relationships = self.relationship_classes[0:3]
        self.spatial_relationships = self.relationship_classes[3:9]
        self.contacting_relationships = self.relationship_classes[9:]


        print('-------loading annotations---------slowly-----------')

        #object_bbox_and_relationship_filtersmall.pkl = object_bbox_and_relationship.pkl
        if filter_small_box:#SGdet-->true
            with open(root_path + '/annotations/person_bbox.pkl', 'rb') as f:
                person_bbox = pickle.load(f)
            f.close()
            with open('/media/jocker/disk2/AG/object_bbox_and_relationship_filtersmall.pkl', 'rb') as f:
                object_bbox = pickle.load(f)
            f.close()
            #objext_bbox={...,'50N4E.mp4/000682.png':'[{'class': 'light',
            #                                                   'bbox': None,
            #                                                   'attention_relationship': None,
            #                                                   'spatial_relationship': None,
            #                                                   'contacting_relationship': None,
            #                                                   'metadata': {'tag': '50N4E.mp4/light/000682', 'set': 'train'},
            #                                                   'visible': False},
            #                                          {'class': 'dish',
            #                                                    'bbox': None,
            #                                                    'attention_relationship': None,
            #                                                    'spatial_relationship': None,
            #                                                    'contacting_relationship': None,
            #                                                    'metadata': {'tag': '50N4E.mp4/dish/000682', 'set': 'train'},
            #                                                    'visible': False}]',...
            #             }
        else:
            with open(root_path + '/annotations/person_bbox.pkl', 'rb') as f:
                person_bbox = pickle.load(f)
                #person_box contains all information of frames which make up the related video
                #eg:...ZYJJF.mp4/000332.png', 'ZYJJF.mp4/000449.png', ...
            f.close()
            with open(root_path+'/annotations/object_bbox_and_relationship.pkl', 'rb') as f:
                object_bbox = pickle.load(f)
            f.close()
        print('--------------------finish!-------------------------')
        #person_box is one-to-one correspondence to object_box  288782
        if datasize == 'mini':
            small_person = {}
            small_object = {}
            for i in list(person_bbox.keys())[:80000]:
                small_person[i] = person_bbox[i]
                small_object[i] = object_bbox[i]
            person_bbox = small_person
            object_bbox = small_object


        # collect valid frames
        video_dict = {}
        for i in person_bbox.keys():
            #print(i)

            #for predcls:
            # person_box contains all information of frames which make up the related video
            #i=keys:ZZXQF.mp4/000628.png...
            #{'bbox': array([[ 24.29774 ,  71.443954, 259.23602 , 268.20288 ]], dtype=float32),
            #'bbox_score': array([0.9960979], dtype=float32),
            # 'bbox_size': (480, 270),
            # 'bbox_mode': 'xyxy',
            #'keypoints': array([[[149.51952 , 120.54931 , 1.],
            #                     [146.48587, 111.43697, 1.],
            #                     [141.09274, 115.824394, 1.],
            #                     [111.76759, 123.58676, 1.],
            #                     [112.44173, 124.26174, 1.],
            #                     [82.10537, 154.6362, 1.],
            #                     [113.45295, 168.47343, 1.],
            #                     [153.56436, 207.96022, 1.],
            #                     [162.66527, 247.44699, 1.],
            #                     [146.48587, 149.91127, 1.],
            #                     [216.59659, 229.22232, 1.],
            #                     [112.10466, 243.73456, 1.],
            #                     [163.3394, 267.69662, 1.],
            #                     [237.83205, 202.56032, 1.],
            #                     [239.18031, 202.56032, 1.],
            #                     [186.93436, 219.0975, 1.],
            #                     [220.9785, 227.87234, 1.]]], dtype = float32)
            #'keypoints_logits': array([[11.073427  , 10.578527  , 10.863391  ,  3.6263876 , 11.451177  ,
            #                            4.500312  ,  6.419147  ,  3.4865067 ,  7.920906  ,  5.6766253 ,
            #                            9.343614  , -0.7024717 , -0.36381796,  1.039403  ,  1.1701871 ,
            #                            -0.03817523, -2.2913933 ]], dtype=float32)}
            #when i=001YG.mp4/000089.png
            #object_bbox[i]=[{'class': 'table',
            # 'bbox': (222.10317460317458, 143.829365079365, 257.77777777777777, 101.11111111111109),
            # 'attention_relationship': ['unsure'],
            # 'spatial_relationship': ['in_front_of'],
            # 'contacting_relationship': ['not_contacting'],
            # 'metadata': {'tag': '001YG.mp4/table/000089', 'set': 'train'},
            # 'visible': True},
            # {'class': 'chair',
            # 'bbox': (56.34126984126985, 179.16666666666663, 192.77777777777777, 90.56890211160798),
            # 'attention_relationship': ['not_looking_at'],
            # 'spatial_relationship': ['beneath', 'behind'],
            # 'contacting_relationship': ['sitting_on', 'leaning_on'],
            # 'metadata': {'tag': '001YG.mp4/chair/000089', 'set': 'train'},
            # 'visible': True}]
            # This picture contains two dict in object_bbox,but they are in one big list.

            #for sgdet:
            #i=001YG.mp4/000089.png
            #the attrabute in object_bbox_and_relationship_filtersmall.pkl is the same as that in object_bbox_and_relationship.pkl
            if object_bbox[i][0]['metadata']['set'] == mode: #train or testing?
                frame_valid = False
                for j in object_bbox[i]: # the frame is valid if there is visible bbox
                    if j['visible']:
                        frame_valid = True
                if frame_valid:
                    video_name, frame_num = i.split('/')
                    if video_name in video_dict.keys():
                        video_dict[video_name].append(i)
                    else:
                        video_dict[video_name] = [i]
        #all_len(video_dict=1810)
        #video_dict is consist of videos which have visible frames
        #eg:{...,ZUU8W.mp4:['ZUU8W.mp4/000091.png', 'ZUU8W.mp4/000270.png', 'ZUU8W.mp4/000450.png', 'ZUU8W.mp4/000630.png', 'ZUU8W.mp4/000809.png',...]
        self.video_list = []
        self.video_size = [] # (w,h)
        self.gt_annotations = []
        self.gt_pose=[]
        self.non_gt_human_nums = 0
        self.non_heatmap_nums = 0
        self.non_person_video = 0
        self.one_frame_video = 0
        self.valid_nums = 0

        '''
        filter_nonperson_box_frame = True (default): according to the stanford method, remove the frames without person box both for training and testing
        filter_nonperson_box_frame = False: still use the frames without person box, FasterRCNN may find the person
        '''
        for i in video_dict.keys():
            #i=ZUU8W.mp4
            video = []
            gt_annotation_video = []
            gt_annotation_pose=[]
            for j in video_dict[i]:
                #j=ZUU8W.mp4/000091.png...
                if filter_nonperson_box_frame:#true
                    if person_bbox[j]['bbox'].shape[0] == 0:
                        self.non_gt_human_nums += 1
                        continue
                    else:
                        video.append(j)
                        #video=[...,ZUU8W.mp4/000091.png,...],ZUU8W.mp4这个文件夹下符合条件的图片汇集成video
                        self.valid_nums += 1
                #extract related person_box from the pkl
                gt_annotation_frame = [{'person_bbox': person_bbox[j]['bbox']}]
                gt_frame_pose=[{'pose_17':person_bbox[j]['keypoints']}]
                # each frames's objects and human
                for k in object_bbox[j]:
                    if k['visible']:
                        assert k['bbox'] != None, 'warning! The object is visible without bbox'
                        k['class'] = self.object_classes.index(k['class'])
                        k['bbox'] = np.array([k['bbox'][0], k['bbox'][1], k['bbox'][0]+k['bbox'][2], k['bbox'][1]+k['bbox'][3]]) # from xywh to xyxy
                        k['attention_relationship'] = torch.tensor([self.attention_relationships.index(r) for r in k['attention_relationship']], dtype=torch.long)
                        # for j=001YG.mp4/000089.png,
                        #k={'class': 'table', 'bbox': (222.10317460317458, 143.829365079365, 257.77777777777777, 101.11111111111109), 'attention_relationship': ['unsure'], 'spatial_relationship': ['in_front_of'], 'contacting_relationship': ['not_contacting'], 'metadata': {'tag': '001YG.mp4/table/000089', 'set': 'train'}, 'visible': True}
                        #and {'class': 'chair', 'bbox': (56.34126984126985, 179.16666666666663, 192.77777777777777, 90.56890211160798), 'attention_relationship': ['not_looking_at'], 'spatial_relationship': ['beneath', 'behind'], 'contacting_relationship': ['sitting_on', 'leaning_on'], 'metadata': {'tag': '001YG.mp4/chair/000089', 'set': 'train'}, 'visible': True}
                        #attention_relationships.index(r) means:what place the ['unsure'] and ['not_looking_at'] locate at in attention_relationships
                        #unsure:2,not_looking_at:1
                        k['spatial_relationship'] = torch.tensor([self.spatial_relationships.index(r) for r in k['spatial_relationship']], dtype=torch.long)
                        k['contacting_relationship'] = torch.tensor([self.contacting_relationships.index(r) for r in k['contacting_relationship']], dtype=torch.long)
                        gt_annotation_frame.append(k)
                        #gt_annotation_frame=[{'person_bbox': person_bbox['001YG.mp4/000089.png']['bbox']},
                        #                     {class,bbox,attention_relationship,...}]
                gt_annotation_video.append(gt_annotation_frame)
                gt_annotation_pose.append(gt_frame_pose)
                #gt_annotation_video=[...,[{person_bbox},
                #                           {class,bbox,attention_relationship,attention_relationship,spatial_relationship,
                #                           contacting_relationship,metadata,
                #                           visible}],
                #                           [{},{}],...]
            if len(video) > 2:
                # if the num of frames(picture,like:ZUU8W.mp4/000091.png) in the video which is a list above 2
                self.video_list.append(video)
                #不同的文件夹组成video_list
                #video_list[0]:只要是帧包含了人物框，就被收入video_list里面
                #[['00607.mp4/000016.png', '00607.mp4/000047.png', '00607.mp4/000077.png', '00607.mp4/000082.png',
                #                   '00607.mp4/000098.png', '00607.mp4/000107.png', '00607.mp4/000114.png',
                #                   '00607.mp4/000129.png', '00607.mp4/000138.png', '00607.mp4/000145.png',
                #                   '00607.mp4/000168.png', '00607.mp4/000187.png', '00607.mp4/000195.png',
                #                   '00607.mp4/000199.png', '00607.mp4/000205.png', '00607.mp4/000219.png',
                #                   '00607.mp4/000223.png', '00607.mp4/000242.png', '00607.mp4/000261.png',
                #                   '00607.mp4/000262.png', '00607.mp4/000268.png', '00607.mp4/000275.png',
                #                   '00607.mp4/000285.png', '00607.mp4/000289.png', '00607.mp4/000302.png',
                #                   '00607.mp4/000316.png', '00607.mp4/000326.png', '00607.mp4/000389.png',
                #                   '00607.mp4/000413.png', '00607.mp4/000418.png', '00607.mp4/000442.png',
                #                   '00607.mp4/000452.png', '00607.mp4/000461.png', '00607.mp4/000480.png',
                #                   '00607.mp4/000484.png', '00607.mp4/000486.png', '00607.mp4/000499.png',
                #                   '00607.mp4/000518.png']]
                self.video_size.append(person_bbox[j]['bbox_size'])
                self.gt_annotations.append(gt_annotation_video)
                self.gt_pose.append(gt_annotation_pose)
                #gt_annotations里是整个文件夹里所有图片的{person_box,object_box}
                #gt_annotations:[...,[...,[{person_box},{class,bbox,attention_relationship,...}],...],...]
                #                    -----------------gt_annotation_video----------------------------
                #                         ------------gt_annotation_frame-----------------------
                #[...[[{'person_bbox': array([[140.7502, 150.3986, 229.9369, 328.0558]], dtype=float32)},
                # {'class': 32, 'bbox': array([208.2368, 229.0974, 270.    , 333.3916]),
                #           'attention_relationship': tensor([0]),
                #           'spatial_relationship': tensor([2]),
                #           'contacting_relationship': tensor([8]),
                #           'metadata': {'tag': '00607.mp4/table/000016', 'set': 'test'},
                #           'visible': True}],
                #[{'person_bbox': array([[158.8245, 131.2105, 255.9878, 354.1025]], dtype=float32)},
                #           {'class': 32, 'bbox': array([208.4464, 239.7143, 269.8031, 348.7546]),
                #           'attention_relationship': tensor([2]),
                #           'spatial_relationship': tensor([2]),
                #           'contacting_relationship': tensor([8]),
                #           'metadata': {'tag': '00607.mp4/table/000047', 'set': 'test'},
                #           'visible': True}],
                #[{'person_bbox': array([[160.8105, 213.7097, 251.6256, 356.7066]], dtype=float32)},
                #            {'class': 32, 'bbox': array([215.9343, 217.8608, 270.    , 350.1521]),
                #            'attention_relationship': tensor([1]),
                #            'spatial_relationship': tensor([0]),
                #            'contacting_relationship': tensor([12]),
                #            'metadata': {'tag': '00607.mp4/table/000077', 'set': 'test'},
                #            'visible': True}]]...]
            elif len(video) == 1:
                self.one_frame_video += 1
            else:
                self.non_person_video += 1

        print('x'*60)
        if filter_nonperson_box_frame:
            print('There are {} videos and {} valid frames'.format(len(self.video_list), self.valid_nums))
            print('{} videos are invalid (no person), remove them'.format(self.non_person_video))
            print('{} videos are invalid (only one frame), remove them'.format(self.one_frame_video))
            print('{} frames have no human bbox in GT, remove them!'.format(self.non_gt_human_nums))
        else:
            print('There are {} videos and {} valid frames'.format(len(self.video_list), self.valid_nums))
            print('{} frames have no human bbox in GT'.format(self.non_gt_human_nums))
            print('Removed {} of them without joint heatmaps which means FasterRCNN also cannot find the human'.format(self.non_heatmap_nums))
        print('x' * 60)

    def __getitem__(self, index):
        #index 是从0开始的索引，就是video_list的索引，文件夹的索引号
        frame_names = self.video_list[index]

        #frame_names=[['0SK6H.mp4/000095.png'],['0SK6H.mp4/000284.png'],['0SK6H.mp4/000473.png'],['0SK6H.mp4/000662.png'],['0SK6H.mp4/000851.png']]
        # video_list[0]:即第一个文件夹
        # [['00607.mp4/000016.png', '00607.mp4/000047.png', '00607.mp4/000077.png', '00607.mp4/000082.png',
        #                   '00607.mp4/000098.png', '00607.mp4/000107.png', '00607.mp4/000114.png',
        #                   '00607.mp4/000129.png', '00607.mp4/000138.png', '00607.mp4/000145.png',
        #                   '00607.mp4/000168.png', '00607.mp4/000187.png', '00607.mp4/000195.png',
        #                   '00607.mp4/000199.png', '00607.mp4/000205.png', '00607.mp4/000219.png',
        #                   '00607.mp4/000223.png', '00607.mp4/000242.png', '00607.mp4/000261.png',
        #                   '00607.mp4/000262.png', '00607.mp4/000268.png', '00607.mp4/000275.png',
        #                   '00607.mp4/000285.png', '00607.mp4/000289.png', '00607.mp4/000302.png',
        #                   '00607.mp4/000316.png', '00607.mp4/000326.png', '00607.mp4/000389.png',
        #                   '00607.mp4/000413.png', '00607.mp4/000418.png', '00607.mp4/000442.png',
        #                   '00607.mp4/000452.png', '00607.mp4/000461.png', '00607.mp4/000480.png',
        #                   '00607.mp4/000484.png', '00607.mp4/000486.png', '00607.mp4/000499.png',
        #                   '00607.mp4/000518.png']]
        #print(frame_names[0])-->00607.mp4/000016.png
        processed_ims = []
        im_scales = []
        origin_img=[]
        gt_framepose=[]
        for idx, name in enumerate(frame_names):
            #对于test：
            #第一个文件夹里的图片
            #idx=0,print(name)-->00607.mp4/000016.png
            #对于train:
            #随机抽取7264个文件夹
            im = imageio.imread(os.path.join(self.frames_path, name)) # channel h,w,3
            h=im.shape[0]
            l=im.shape[1]
            # self.frame_path+name = '/media/wow/disk2/AG/dataset/frames/00607.mp4/000016.png'
            #print(im.shape)-->(480, 270, 3)
            im = im[:, :, ::-1] # rgb -> bgr
            #print(im.shape)-->(480, 270, 3)
            im, im_scale = prep_im_for_blob(im, [[[102.9801, 115.9465, 122.7717]]], 600, 1000)
            #cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
            #(1067, 600, 3),2.2
            im_scales.append(im_scale)#缩放的倍数
            processed_ims.append(im)#处理好的图片
            origin_img.append([h,l])
        #到此processed_ims和im_scales里存放着当前文件夹下，互相对应的图片和缩放倍率
        blob = im_list_to_blob(processed_ims)#shape-->[38,1067,600,3],38张图片
        im_info = np.array([[blob.shape[1], blob.shape[2], im_scales[0]]],dtype=np.float32)
        #im_info=np.arrary([[1067,600,2.2]])
        im_info = torch.from_numpy(im_info).repeat(blob.shape[0], 1)#(38,1)
        #print(im_info)--->[[1067.0000,   600.000,   2.2222]
        #                   [1067.0000,   600.000,   2.2222]
        #                                 ...
        #                   [1067.0000,   600.000,   2.2222]}
        #print(im_info.shape)-->torch.size([38,3])
        img_tensor = torch.from_numpy(blob)
        img_tensor = img_tensor.permute(0, 3, 1, 2)
        #shape-->[38,3,1067,600]

        gt_boxes = torch.zeros([img_tensor.shape[0], 1, 5])
        #print(gt_boxes.shape)-->torch.Size([38, 1, 5]),38个1行5列向量
        num_boxes = torch.zeros([img_tensor.shape[0]], dtype=torch.int64)
        #print(num_boxes.shape)-->torch.Size([38]),[0,0,0,0,0...]
        openpose_frame=self.video_list[index]

        return img_tensor, im_info, gt_boxes, num_boxes, index, openpose_frame, origin_img, gt_framepose,frame_names
        #img_tensor:对img_info转换后图像的张量信息,shape-->[38,3,1067,600]
        #img_info:原图像格式信息,print(im_info.shape)-->torch.size([38,3])
        #gt_boxes:标记框,torch.Size([38, 1, 5]),38个1行5列向量
        #num_boxes:每个图片内有几个框,torch.Size([38]),[0,0,0,0,0...]
        #index:文件夹的索引号,0,1,2...

    def __len__(self):
        return len(self.video_list)

def cuda_collate_fn(batch):
    """
    don't need to zip the tensor

    """
    return batch[0]

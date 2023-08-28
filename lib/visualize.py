import torch
import cv2
import numpy


object = ['person', 'bag', 'bed','blanket', 'book', 'box', 'broom', 'chair', 'closet/cabinet', 'clothes',
          'cup/glass/bottle', 'dish', 'door', 'doorknob', 'doorway', 'floor', 'food', 'groceries',
          'laptop', 'light', 'medicine', 'mirror', 'papertonebook', 'phone/camera', 'picture',
          'pillow', 'refrigerator', 'sandwich', 'shelf', 'shoe', 'sofacouch', 'table', 'television',
          'towel', 'vacuum', 'window']

relations = ['look at', 'not looking at', 'unsure', 'above', 'beneath', 'in front of', 'behind', 'on the side of',
             'in', 'carrying', 'covered by', 'drinking from', 'eating', 'having it on the back', 'holding',
             'leaning on', 'lying on', 'not contacting', 'other relationship', 'sitting on', 'standing on',
             'touching', 'twisting', 'wearing', 'wiping', 'writing on']


def visualize(pred,video_len,frame,scale,pred_bbox):
    #print(frame)
    # ['00607.mp4/000016.png', '00607.mp4/000047.png', '00607.mp4/000077.png', '00607.mp4/000082.png',
    #  '00607.mp4/000098.png', '00607.mp4/000107.png', '00607.mp4/000114.png', '00607.mp4/000129.png',
    #  '00607.mp4/000138.png', '00607.mp4/000145.png', '00607.mp4/000168.png', '00607.mp4/000187.png',
    #  '00607.mp4/000195.png', '00607.mp4/000199.png', '00607.mp4/000205.png', '00607.mp4/000219.png',
    #  '00607.mp4/000223.png', '00607.mp4/000242.png', '00607.mp4/000261.png', '00607.mp4/000262.png',
    #  '00607.mp4/000268.png', '00607.mp4/000275.png', '00607.mp4/000285.png', '00607.mp4/000289.png',
    #  '00607.mp4/000302.png', '00607.mp4/000316.png', '00607.mp4/000326.png', '00607.mp4/000389.png',
    #  '00607.mp4/000413.png', '00607.mp4/000418.png', '00607.mp4/000442.png', '00607.mp4/000452.png',
    #  '00607.mp4/000461.png', '00607.mp4/000480.png', '00607.mp4/000484.png', '00607.mp4/000486.png',
    #  '00607.mp4/000499.png', '00607.mp4/000518.png']

    # print(pred)
    # [array([[0, 1, 1, 7, 17],
    #         [0, 2, 7, 1, 5],
    #         [0, 3, 16, 1, 4],
    #         [0, 4, 1, 16, 20],
    #         [0, 5, 1, 32, 17],
    #         [0, 6, 32, 1, 5],
    #         [0, 7, 1, 32, 1],
    #         [0, 8, 1, 29, 17],
    #         [0, 9, 29, 1, 7],
    #         [1, 0, 1, 29, 1],
    #         [2, 0, 34, 1, 5],
    #         [3, 0, 1, 34, 14],
    #         [4, 0, 1, 34, 0],
    #         [5, 0, 12, 1, 5],
    #         [6, 0, 1, 12, 17],
    #         [7, 0, 1, 7, 0],
    #         [8, 0, 1, 16, 0],
    #         [9, 0, 12, 1, 5],
    #         [0, 1, 1, 12, 17],
    #         [0, 2, 12, 1, 5],
    #         [0, 3, 1, 12, 17],
    #         [0, 4, 17, 1, 5],
    #         [0, 5, 1, 17, 17],
    #         [0, 6, 1, 12, 0],
    #         [0, 7, 1, 12, 0],
    #         [0, 8, 1, 12, 0],
    #         [0, 9, 1, 17, 1]]), array([[10, 11, 1, 7, 17],
    #                                    [10, 12, 7, 1, 7],
    #                                    [10, 13, 16, 1, 4],
    #                                    [10, 14, 1, 16, 20],
    #                                    [10, 15, 32, 1, 5],
    #                                    [10, 16, 1, 32, 17],
    #                                    [11, 10, 1, 32, 1],
    #                                    [12, 10, 1, 29, 17],
    #                                    [13, 10, 29, 1, 7],
    #                                    [14, 10, 1, 29, 1],
    #                                    [15, 10, 1, 16, 0],
    #                                    [16, 10, 1, 7, 1],
    #                                    [10, 11, 34, 1, 5],
    #                                    [10, 12, 1, 34, 14],
    #                                    [10, 13, 1, 34, 0],
    #                                    [10, 14, 12, 1, 5],
    #                                    [10, 15, 1, 12, 14],
    #                                    [10, 16, 1, 12, 0]]), array([[17, 18, 16, 1, 4],
    #                                                                 [17, 19, 1, 7, 17],
    #                                                                 [17, 20, 7, 1, 7],
    #                                                                 ...


    #print(pred_bbox[:12,:])
    # [[0.     140.5056 147.3516 229.6832 326.5945]
    #  [0.      31.9522 333.1364 148.0413 359.312]
    #  [0.     223.3091 220.2276 251.226 230.2844]
    #  [0.     223.1333 224.8624 250.1424 235.2108]
    #  [0.     222.1688 228.2177 249.8033 242.5689]4
    #  [0.       0.     265.8343 267.5671 479.7]
    #  [0.     230.8789 229.222 250.3605 242.0099]
    #  [0.    19.9209 206.3232 122.7106 329.6276]
    #  [0.    201.2594 214.7666 269.55 332.676]
    #  [0.    198.4026 218.5288 225.3175 235.4141]
    #  [1.    157.6989 128.8508 254.6045 354.6502]
    #  [1.       6.1447 350.9258 139.7769 380.6629]]

    for i in range(video_len):
        ind=0
        path='/media/wow/disk2/AG/dataset/frames/'+frame[i]
        image = cv2.imread(path)
        #image = cv2.resize(image, None, None, fx=float(scale), fy=float(scale), interpolation=cv2.INTER_LINEAR)

        frame_box_num=len(pred[i])        #print(frame_box_num)#27
        per_frame_box_num=frame_box_num/3
        classA_bbox_index=pred[i][:,0]
        classB_bbox_index=pred[i][:,1]
        classA=pred[i][:,2]
        classB=pred[i][:,3]
        relat=pred[i][:,4]

        cv2.rectangle(image,(230,229),(250,242),(0,0,255),2)#object
        cv2.putText(image,object[classB[4]-1],(0,479),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,200,0),1)
        cv2.rectangle(image,(140,147),(229,326),(0,0,255),2)#person
        cv2.putText(image,object[classA[4]-1],(140,326),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,200,0),1)
        #第4行[0,5,1,32,17],坐标框为0和5,物体类为第1-1和32-1,关系为17-1  [0  6 32  1  5]
        #cv2.putText(image,relations[relat])
        #cv2.rectangle(image,(pred_bbox[ind:per_frame_box_num,:][classA_bbox_index],pred_bbox[ind:per_frame_box_num,:][classB_bbox_index]), (pred[classB_bbox_index],), (0, 0, 255), 2)
        #print(classA_bbox_index)[0 0 0 0 0 0 0 0 0 1 2 3 4 5 6 7 8 9 0 0 0 0 0 0 0 0 0]
        #print(classB_bbox_index)[1 2 3 4 5 6 7 8 9 0 0 0 0 0 0 0 0 0 1 2 3 4 5 6 7 8 9]
        #print(classA)[ 1 7  16  1  1  32   1   1   29  1   34  1   1   12  1   1  1   12  1   12  1   17  1   1  1  1  1]
        #print(classB)[ 7  1  1 16  32  1   32  29  1   29  1   34  34  1   12  7  16  1   12  1   12  1   17  12 12 12 17]
        #print(relat) [17  5  4 20  17  5   1   17  7   1   5   14  0   5   17  0  0   5   17  5   17  5   17  0  0  0  1]

        cv2.imshow('pic', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit()
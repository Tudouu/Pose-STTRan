import os
import pickle
import numpy as np
import cv2
root_path='/media/wow/disk2/AG/dataset'
"""with open('/media/wow/disk2/AG/object_bbox_and_relationship_filtersmall.pkl', 'rb') as f:
    object_bbox1 = pickle.load(f)
f.close()
with open(root_path + '/annotations/object_bbox_and_relationship.pkl', 'rb') as f:
    object_bbox2 = pickle.load(f)
f.close()"""
with open('/media/wow/disk2/AG/dataset/annotations/person_bbox.pkl','rb') as k:
    person_bbox=pickle.load(k)
k.close()

root="/media/wow/disk2/AG/dataset/frames/"
values='1BVUA.mp4/000474.png'
picture=cv2.imread(root+values)
a=person_bbox[values]["keypoints"][:,:,:2]
for i in range(a.shape[1]):
    cv2.circle(picture,(int(a[:,i,:][0][0]),int(a[:,i,:][0][1])),3,(255,0,0),thickness=-1)
cv2.imshow("image",picture)
cv2.waitKey(0)
cv2.destroyAllWindows()

exit()




print(object_bbox1['0VOQC.mp4/000023.png'])
print(object_bbox1['0VOQC.mp4/000038.png'])
print(object_bbox1['0VOQC.mp4/000042.png'])
print(object_bbox1['0VOQC.mp4/000043.png'])
print(object_bbox1['0VOQC.mp4/000043.png'])
print(object_bbox1['0VOQC.mp4/000044.png'])
print(object_bbox1['0VOQC.mp4/000052.png'])
print(object_bbox1['0VOQC.mp4/000067.png'])
print(object_bbox1['0VOQC.mp4/000069.png'])
print(object_bbox1['0VOQC.mp4/000072.png'])
print(object_bbox1['0VOQC.mp4/000090.png'])
print(object_bbox1['0VOQC.mp4/000096.png'])
print(object_bbox1['0VOQC.mp4/000100.png'])
print(object_bbox1['0VOQC.mp4/000105.png'])
print(object_bbox1['0VOQC.mp4/000117.png'])
print(object_bbox1['0VOQC.mp4/000120.png'])
print(object_bbox1['0VOQC.mp4/000123.png'])
print(object_bbox1['0VOQC.mp4/000127.png'])
print(object_bbox1['0VOQC.mp4/000129.png'])

#print(object_bbox2['50N4E.mp4/000682.png'])
#drinking from
"""print(object_bbox1['0JHMW.mp4/000026.png'])
print(object_bbox1['0JHMW.mp4/000029.png'])
print(object_bbox1['0JHMW.mp4/000076.png'])
print(object_bbox1['0JHMW.mp4/000086.png'])
print(object_bbox1['0JHMW.mp4/000127.png'])
print(object_bbox1['0JHMW.mp4/000143.png'])
print(object_bbox1['0JHMW.mp4/000145.png'])
print(object_bbox1['0JHMW.mp4/000174.png'])
print(object_bbox1['0JHMW.mp4/000177.png'])
print(object_bbox1['0JHMW.mp4/000200.png'])
print(object_bbox1['0JHMW.mp4/000203.png'])
print(object_bbox1['0JHMW.mp4/000227.png'])
print(object_bbox1['0JHMW.mp4/000228.png'])
print(object_bbox1['0JHMW.mp4/000231.png'])
print(object_bbox1['0JHMW.mp4/000232.png'])
print(object_bbox1['0JHMW.mp4/000257.png'])
print(object_bbox1['0JHMW.mp4/000261.png'])
print(object_bbox1['0JHMW.mp4/000285.png'])
print(object_bbox1['0JHMW.mp4/000288.png'])
print(object_bbox1['0JHMW.mp4/000343.png'])
print(object_bbox1['0JHMW.mp4/000345.png'])
print(object_bbox1['0JHMW.mp4/000401.png'])
print(object_bbox1['0JHMW.mp4/000402.png'])"""
#print(person_bbox['0CFQV.mp4/000020.png'])
"""path = '/media/wow/disk2/AG/dataset/frames/00607.mp4/000016.png'
a=np.array([[0, 140, 147, 229, 326],
            [0,  31, 333, 148, 359],
            [0, 223, 220, 251, 230],
            [0, 225, 223, 253, 236]])
img = cv2.imread(path)
for i in a:
    cv2.rectangle(img, (i[1], i[2]), (i[3], i[4]), (255, 0, 0), 1)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

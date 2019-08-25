import numpy as np
import glob
import os
from PIL import Image
from operator import itemgetter
dir = '/home/shared/datasets/cars.merged.new/train_A/'
base_image_path = os.path.join(dir,'002445.png')
base_image = np.array(Image.open(base_image_path))
base_image[base_image==255] = 1
name_l = []
IoU = []
for name in glob.glob(os.path.join(dir,'*')):
    compare_image = np.array(Image.open(name))
    compare_image[compare_image==255]=1
    intersection =  np.sum(base_image*compare_image)
    union = np.sum(np.ceil((base_image + compare_image)/2))
    IoU.append(intersection/union)
    name_l.append(name)
list_1 , list_2 = [list(x) for x in zip(*sorted(zip(IoU, name_l), key=itemgetter(0),reverse=True))]
print(list_1[10:20])
print(list_2[10:20])

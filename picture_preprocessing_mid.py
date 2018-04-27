from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2 as cv
import crop
from pylab import *
import numpy as np
import csv
import time


dr1 = 'E:/carry/Download/DR_DATASET/DR_data/train/'
dr2 = 'E:/carry/Download/DR_DATASET/DR_data/test_radius/'
dr3 = 'E:/carry/Download/DR_DATASET/DR_data/test_crop_256/'
dir_trainLabel = 'E:/carry/Download/DR_DATASET/DR_data/testLabels.csv'

image_names = []
with open(dir_trainLabel, mode='r', encoding='utf-8') as csvfile:
    img_name_reader = csv.reader(csvfile)
    for row in img_name_reader:
        image_names.append(row[0]+'.jpeg')

#print(image_names[32681])
#image_names = image_names[:10000]
image_names = image_names[30556:32681]
#image_names = image_names[32682:38556]

num = len(image_names)
i = 0

for i,image_name in enumerate(image_names):

    fname2 = dr2 + image_name
    fname3 = dr3 + image_name

    img_resized = crop.convert(fname2, 256)
    img_resized.save(fname3)

    if i%10 == 0:
        print(time.strftime('%Y-%m-%d %H:%M:%S  ',time.localtime(time.time()))
              + '图片处理进度：%.5f %%' % ((i / num)*100) + '   i = %d'% i)


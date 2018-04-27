from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2 as cv
import crop
from pylab import *
import numpy as np
import csv
import time


dr1 = 'E:/carry/Download/DR_DATASET/DR_data/test/'
dr2 = 'E:/carry/Download/DR_DATASET/DR_data/test_radius/'
dr3 = 'E:/carry/Download/DR_DATASET/DR_data/test_crop_512/'
dir_testLabel = 'E:/carry/Download/DR_DATASET/DR_data/testLabels.csv'

image_names = []
with open(dir_testLabel, mode='r', encoding='utf-8') as csvfile:
    img_name_reader = csv.reader(csvfile)
    for row in img_name_reader:
        image_names.append(row[0]+'.jpeg')

# print(image_names[32682])


image_names = image_names[32682:38556]
num = len(image_names)
i = 1

for i,image_name in enumerate(image_names):

    fname1 = dr1 + image_name
    fname2 = dr2 + image_name
    fname3 = dr3 + image_name
    scale = 500
    img_scale = crop.scaleRadius(fname1, scale)
    img = cv.addWeighted(img_scale, 4, cv.GaussianBlur(img_scale, (0, 0), scale/30), -4, 128)
    img_z = np.zeros(img.shape)
    cv.circle(img_z, (int(img.shape[1]/2), int(img.shape[0]/2)),
              int(scale*0.9), (1,1,1), -1, 8, 0)
    img = img * img_z +128 * (1 - img_z)
    cv.imwrite(fname2, img)


    img_resized = crop.convert(fname2, 512)
    img_resized.save(fname3)

    if i%10 == 0:
        print(time.strftime('%Y-%m-%d %H:%M:%S  ',time.localtime(time.time()))
              + '图片处理进度：%.5f %%' % ((i / num)*100) + '   i = %d'% i)


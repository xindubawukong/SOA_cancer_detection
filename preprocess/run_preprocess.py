import os
from main import *
import cv2


path = '../../cancer_detection'
num = 0
for cur_dir, dirs, files in os.walk(path):
    for file in files:
        if file[-3:] not in ['jpg', 'bmp', 'png', 'tif']:
            continue
        if file[:5] == 'new1_':
            continue
        print('\nPre-processing: ' + os.path.join(cur_dir, file))
        img = convert_img(os.path.join(cur_dir, file), (512, 512))
        cv2.imwrite(os.path.join(cur_dir, 'new1_' + file), img)
        print('Saved to: ' + os.path.join(cur_dir, 'new1_' + file))
        num += 1
        print('num = %d' % num)
print(num)
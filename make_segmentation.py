import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import numpy as np

import os, shutil

# 0 - Background
# 1 - Hair
# 2 - Face
# 3 - Top-Body-Skin
# 4 - Top-Body-Clothes
# 5 - Lower-Body-Skin
# 6 - Lower-Body-Clothes
# 7 - Upper-Accessories
# 8 - Lower-Accessories
# palette (color map) describes the (R, G, B): Label pair
# palette = {(0,   0,   0) : 0 ,
#          (1,  1, 1) : 7,
#          (2,  2, 2) : 1,
#          (3,  3, 3) : 7,
#          (4,  4, 4) : 7,
#          (5,  5, 5) : 4,
#          (6,  6, 6) : 4,
#          (7,  7, 7) : 4,
#          (8,  8, 8) : 8,
#          (9,  9, 9) : 6,
#          (10,  10, 10) : 3,
#          (11,  11, 11) : 7,
#          (12,  12, 12) : 4,
#          (13,  13, 13) : 2,
#          (14,  14, 14) : 3,
#          (15,  15, 15) : 3,
#          (16,  16, 16) : 5,
#          (17,  17, 17) : 5,
#          (18,  18, 18) : 8,
#          (19,  19, 19) : 8
#          }


# 0 - Background
# 1 - Hair
# 2 - Face
# 3 - Top-Body-Skin
# 4 - Top-Body-Clothes
# 5->3 - Lower-Body-Skin
# 6->4 - Lower-Body-Clothes
# 7->5 - Upper-Accessories
# 8->5 - Lower-Accessories
palette = {(0,   0,   0) : 0 ,
         (1,  1, 1) : 5,
         (2,  2, 2) : 1,
         (3,  3, 3) : 5,
         (4,  4, 4) : 5,
         (5,  5, 5) : 4,
         (6,  6, 6) : 4,
         (7,  7, 7) : 4,
         (8,  8, 8) : 5,
         (9,  9, 9) : 4,
         (10,  10, 10) : 3,
         (11,  11, 11) : 5,
         (12,  12, 12) : 4,
         (13,  13, 13) : 2,
         (14,  14, 14) : 3,
         (15,  15, 15) : 3,
         (16,  16, 16) : 3,
         (17,  17, 17) : 3,
         (18,  18, 18) : 5,
         (19,  19, 19) : 5
         }

def convert_from_color_segmentation(arr_3d):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i
    return arr_2d


label_dir = './mask/'
new_label_dir = './segmentation/'

if not os.path.isdir(new_label_dir):
	print("creating folder: ",new_label_dir)
	os.mkdir(new_label_dir)
else:
	print("Folder alread exists. Delete the folder and re-run the code!!!")


label_files = os.listdir(label_dir)

count = 1
train_list = []
val_list = []
trainval_list = []

for l_f in tqdm(label_files):
    arr = np.array(Image.open(label_dir + l_f))
    arr = arr[:,:,0:3]
    arr_2d = convert_from_color_segmentation(arr)
    Image.fromarray(arr_2d).save(new_label_dir + l_f)

    #print( "New label: ", new_label_dir)
    #print( "LF: ", l_f)
    test = l_f.split(".")
    test2 = str(test[0]).split("_")
    imageNum = test2[-1]
    #print( "TEST: ", test)
    #print( "New lf: ", l_f)
    trainval_list.append("image_" + imageNum)
    if count > 8:
      val_list.append("image_" + imageNum)
      count += 1
    if count <= 8:
      train_list.append("image_" + imageNum)
      count += 1
    if count == 11:
      count = 1

print( len (train_list))
print( len(val_list))
print( len(trainval_list))
print( count)

# define list of places
with open( "./ImageSets/train.txt" , 'w') as filehandle:
  for listitem in train_list:
    filehandle.write('%s\n' % listitem)
with open( "./ImageSets/trainval.txt" , 'w') as filehandle:
  for listitem in trainval_list:
    filehandle.write('%s\n' % listitem)
with open( "./ImageSets/val.txt" , 'w') as filehandle:
  for listitem in val_list:
    filehandle.write('%s\n' % listitem)
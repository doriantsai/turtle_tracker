'''
Moves data downloaded data from Cvat into the correct folders for yolov5 training
(folders need to be set up before hand, and the retrieve path and save path editted)
'''

import glob
import shutil
import random
import os

retrievepath = '/home/raineai/Downloads/obj_train_data'
savepath = '/home/raineai/Turtles/datasets/job11'

imagelist = glob.glob(os.path.join(retrievepath, '*.PNG'))
txtlist = glob.glob(os.path.join(retrievepath, '*.txt'))
txtlist.sort()
imagelist.sort()

validimg = []
validtext = []

for i, text in enumerate(txtlist):#convert all labels to turtles (0)
    with open(text, 'r') as f:
        lines = f.readlines()

    with open(text, 'w') as f:
        for line in lines:
            a,b,c,d,e = line.split()
            if int(a) == 1:
                a = "0"
            f.write(f'{a} {b} {c} {d} {e}\n')

#pick some random files
for i in range(10):
    r = random.randint(0, len(txtlist))
    validimg.append(imagelist[r])
    validtext.append(txtlist[r])
    txtlist.remove(txtlist[r])
    imagelist.remove(imagelist[r])

for i, text in enumerate(txtlist):
    shutil.move(text, savepath+'/train/labels')


for i, img in enumerate(imagelist):
    shutil.move(img, savepath+'/train/images')
    

for i, text in enumerate(validtext):
    shutil.move(text, savepath+'/valid/labels')
    

for i, img in enumerate(validimg):
    shutil.move(img, savepath+'/valid/images')
    

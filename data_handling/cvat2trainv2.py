'''
Moves data downloaded data from Cvat into the correct folders for yolov5 training
(folders need to be set up before hand, and the retrieve path and save path editted)
'''

import glob
import shutil
import random
import os

imgno = 1000
train_ratio = 0.8
test_ratio = 0.1
valid_ratio = 0.1

def check_ratio(test_ratio,train_ratio,valid_ratio):
    if(test_ratio>1 or test_ratio<0): ValueError(test_ratio,f'test_ratio must be > 1 and test_ratio < 0, test_ratio={test_ratio}')
    if(train_ratio>1 or train_ratio<0): ValueError(train_ratio,f'train_ratio must be > 1 and train_ratio < 0, train_ratio={train_ratio}')
    if(valid_ratio>1 or valid_ratio<0): ValueError(valid_ratio,f'valid_ratio must be > 1 and valid_ratio < 0, valid_ratio={valid_ratio}')
    if not((train_ratio+test_ratio+valid_ratio)==1): ValueError("sum of train/val/test ratio must equal 1")
check_ratio(test_ratio,train_ratio,valid_ratio)

retrievepath = os.path.join('/home/raineai/Downloads','obj_train_data')
savepath = os.path.join('/home/raineai/Turtles/datasets','job11')

def clean_dirctory(savepath):
    if os.path.isdir(savepath):
        shutil.rmtree(savepath)
    os.makedirs(savepath)
clean_dirctory(savepath)

imagelist = glob.glob(os.path.join(retrievepath, '*.PNG'))
txtlist = glob.glob(os.path.join(retrievepath, '*.txt'))
txtlist.sort()
imagelist.sort()

validimg = []
validtext = []
testimg = []
testtext = []

for i, text in enumerate(txtlist):#convert all labels to turtles (0)
    with open(text, 'r') as f:
        lines = f.readlines()

    with open(text, 'w') as f:
        for line in lines:
            a,b,c,d,e = line.split()
            if int(a) == 1:
                a = "0"
            f.write(f'{a} {b} {c} {d} {e}\n')

def seperate_files(number,imglist,textlist):
    for i in range(number):
        r = random.randint(0, len(textlist))
        imglist.append(imagelist[r])
        textlist.append(txtlist[r])
        txtlist.remove(txtlist[r])
        imagelist.remove(imagelist[r])

#pick some random files
seperate_files(imgno*valid_ratio,validimg,validtext) #get some valid images
seperate_files(imgno*test_ratio,testimg,testtext)

def move_file(filelist,savepath,second_path):
    for i, item in enumerate(filelist):
        shutil.move(item, os.path.join(savepath,second_path))

move_file(txtlist,savepath,'train/labels')
move_file(imagelist,savepath,'train/images')
move_file(validtext,savepath,'valid/labels')
move_file(validimg,savepath,'valid/images')
move_file(testimg,savepath,'test/labels')
move_file(testtext,savepath,'test/images')
    

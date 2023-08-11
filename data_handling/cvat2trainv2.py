'''
Moves data downloaded data from Cvat into the correct folders for yolov5 training
(folders need to be set up before hand, and the retrieve path and save path editted)
'''

import glob
import shutil
import random
import os

train_ratio = 0.8
test_ratio = 0.1
valid_ratio = 0.1

HOME_DIR = '/home/serena/Data/Turtles/turtle_datasets'
VID_DIR = 'job10_041219-0-1000'
IN_DIR = 'obj_train_data'
IMG_TYPE = '.PNG'
OUT_DIR = 'split_data_2_class'

def check_ratio(test_ratio,train_ratio,valid_ratio):
    if(test_ratio>1 or test_ratio<0): ValueError(test_ratio,f'test_ratio must be > 1 and test_ratio < 0, test_ratio={test_ratio}')
    if(train_ratio>1 or train_ratio<0): ValueError(train_ratio,f'train_ratio must be > 1 and train_ratio < 0, train_ratio={train_ratio}')
    if(valid_ratio>1 or valid_ratio<0): ValueError(valid_ratio,f'valid_ratio must be > 1 and valid_ratio < 0, valid_ratio={valid_ratio}')
    if not((train_ratio+test_ratio+valid_ratio)==1): ValueError("sum of train/val/test ratio must equal 1")
check_ratio(test_ratio,train_ratio,valid_ratio)


retrievepath = os.path.join(HOME_DIR,VID_DIR,IN_DIR)
savepath = os.path.join(HOME_DIR,VID_DIR,OUT_DIR)


def clean_dirctory(savepath):
    if os.path.isdir(savepath):
        shutil.rmtree(savepath)
    os.makedirs(savepath, exist_ok=True)
clean_dirctory(savepath)

imagelist = glob.glob(os.path.join(retrievepath, '*'+IMG_TYPE))
txtlist = glob.glob(os.path.join(retrievepath, '*.txt'))
txtlist.sort()
imagelist.sort()
imgno = len(txtlist) # NOTE not all jobs have 1000 images

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
    for i in range(int(number)):
        r = random.randint(0, len(textlist))
        imglist.append(imagelist[r])
        textlist.append(txtlist[r])
        txtlist.remove(txtlist[r])
        imagelist.remove(imagelist[r])

#pick some random files
seperate_files(imgno*valid_ratio,validimg,validtext) #get some valid images
seperate_files(imgno*test_ratio,testimg,testtext)

# function to preserve symlinks of src file, otherwise default to copy
def copy_link(src, dst):
    if os.path.islink(src):
        linkto = os.readlink(src)
        os.symlink(linkto, os.path.join(dst, os.path.basename(src)))
    else:
        shutil.copy(src, dst)
        

def move_file(filelist,savepath,second_path):
    output_path = os.path.join(savepath, second_path)
    clean_dirctory(output_path)
    os.makedirs(output_path, exist_ok=True)
    for i, item in enumerate(filelist):
        # shutil.move(item, os.path.join(savepath,second_path))
        copy_link(item, output_path)

move_file(txtlist,savepath,'train/labels')
move_file(imagelist,savepath,'train/images')
move_file(validtext,savepath,'valid/labels')
move_file(validimg,savepath,'valid/images')
move_file(testimg,savepath,'test/images')
move_file(testtext,savepath,'test/labels')
    

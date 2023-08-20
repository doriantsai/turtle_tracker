import os
import cv2
import code
import glob
### NOT COMPLETE -- But does run ###
##  Todo:
#	works one job and folder at a time (ie. job10/test), can it be improve
#	control number on turtles should be less dodgy
'''
Python file to create cropped images of painted and non-painted turtles
'''
dataset = 'job11_2clases'
data_dir = os.path.join('/home/raineai/Turtles/datasets',dataset,'split_data/test_old')
pad = 10
imglist = sorted(glob.glob(os.path.join(data_dir, 'images', '*.PNG')))
txtlist = sorted(glob.glob(os.path.join(data_dir, 'labels', '*.txt')))

nopainted = 0
noturtles = 0


for j, filename in enumerate(imglist): #for every file
    #j = 17
    filename = imglist[j]
    img_path = os.path.join(data_dir, filename)
    img = cv2.imread(img_path)
    imgw, imgh = img.shape[1], img.shape[0]

    txt_path = txtlist[j]
    #code.interact(local=dict(globals(),**locals()))    
    x1,y1,x2,y2,i = [],[],[],[],0
    with open(txt_path, 'r') as f: #for every line
        for line in f:                
            if i>5: #painted objects always stored above turtles, don't care after the first few lines
                break
            a,b,c,d,e = line.split()
            w = round(float(b)*imgw) # TODO Dorian: as I read this, this is incorrect
            h = round(float(c)*imgh) # TODO Dorian: [class, x_center_n,  y_center_n, width_n, height_n] https://docs.ultralytics.com/datasets/detect/
            y1.append(h+round(float(e)*imgh/2))
            y2.append(h-round(float(d)*imgh/2))
            x1.append(w+round(float(d)*imgw/2))
            x2.append(w-round(float(d)*imgw/2))
            xmax,xmin,ymax,ymin = max(x1[i],x2[i]),min(x1[i],x2[i]),max(y1[i],y2[i]),min(y1[i],y2[i])
            #creare the crop image
            xminp,yminp = max(xmin-pad,1), max(ymin-pad,1)
            obj_img = img[yminp:ymax+pad,xminp:xmax+pad] 
            if obj_img.size == 0: #incase of weird outputs - should no longer happen
                #obj_img = img[ymin:ymax,xmin:xmax]
                #code.interact(local=dict(globals(),**locals()))
                print("Weird outputs at:")
                print(j)
                print(filename)

            if a == '1': #if its painted, save
                nopainted = nopainted + 1
                classlabel = 'painted'
                  
            else:
                noturtles = noturtles + 1
                classlabel = 'turtle'            
            
            class_dir = os.path.join(data_dir, classlabel)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            obj_filename =  filename.replace(os.path.join(data_dir,'images/'), '')[:-4]
            obj_filename = dataset+"_"+obj_filename
            obj_path = os.path.join(class_dir, obj_filename+str(i)+'.PNG')
            #code.interact(local=dict(globals(),**locals()))
            
            if classlabel=='painted':
                cv2.imwrite(obj_path, obj_img)
            elif noturtles-50<nopainted:
                cv2.imwrite(obj_path, obj_img)

            #code.interact(local=dict(globals(),**locals()))
            i += 1
                        

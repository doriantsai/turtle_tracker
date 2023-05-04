import os
import cv2
import code
import glob
### NOT COMPLETE -- But does run ###
##  Todo:
#	for x1,x2,y1,y1:
#		make sure crop is going from ymin:ymax & xmin:xmax, then add padding
#		Why were values going negative? Check calculations
#		make sure padding doesn't go negative / set varible max(varible, 1)
#	works one job and folder at a time (ie. job10/test), can it be improved
#	files created (croped images) should include job name (so can integrate with other job data)
#	remove class_dict -> only had for debugging
#	add way to have same number of turtles as painted / other control number on turtles
'''
Python file to create cropped images of painted and non-painted turtles
'''
data_dir = '/home/raineai/Turtles/datasets/job11_2clases/split_data/test'
pad = 5
class_dict = {}

imglist = sorted(glob.glob(os.path.join(data_dir, 'images', '*.PNG')))
txtlist = sorted(glob.glob(os.path.join(data_dir, 'labels', '*.txt')))


for j, filename in enumerate(imglist): #for every file
    filename = imglist[j]
    img_path = os.path.join(data_dir, filename)
    img = cv2.imread(img_path)
    imgw, imgh = img.shape[1], img.shape[0]

    txt_path = txtlist[j]
    #code.interact(local=dict(globals(),**locals())    
    x1,y1,x2,y2,i = [],[],[],[],0
    with open(txt_path, 'r') as f: #for every line
        for line in f:                
            if i>5: #painted objects always stored above turtles, don't care after the first few lines
                break
            a,b,c,d,e = line.split()
            w = round(float(b)*imgw)
            h = round(float(c)*imgh)
            y1.append(h+round(float(e)*imgh/2))
            y2.append(h-round(float(d)*imgh/2))
            x1.append(w+round(float(d)*imgw/2))
            x2.append(w-round(float(d)*imgw/2))
            #creare the crop image
            obj_img = img[y2[i]-pad:y1[i]+pad,x2[i]-pad:x1[i]+pad]
            if obj_img.size == 0: #incase of weird outputs
                y2[i],y1[i],x2[i],x1[i] = max(y2[i],1),max(y1[i],1),max(x2[i],1),max(x1[i],1)
                obj_img = img[y2[i]:y1[i],x2[i]:x1[i]]
        

            if a == '1': #if its painted, save
                classlabel = 'painted'
                class_dir = os.path.join(data_dir, classlabel)
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)

                obj_filename =  filename.replace(os.path.join(data_dir,'images/'), '')[:-4]
                obj_path = os.path.join(class_dir, obj_filename+str(i)+'.PNG')
                #code.interact(local=dict(globals(),**locals()))
                cv2.imwrite(obj_path, obj_img)

            else:
                classlabel = 'turtle'
                #can add a percentage chance, as there will be more turtles then painted
                class_dir = os.path.join(data_dir, classlabel)
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)
                obj_filename =  filename.replace(os.path.join(data_dir,'images/'), '')[:-4]
                obj_path = os.path.join(class_dir, obj_filename+str(i)+'.PNG')
                #cv2.imwrite(obj_path, obj_img)
        
            #update the dictionary storing
            if classlabel not in class_dict:
                class_dict[classlabel] = []
            class_dict[classlabel].append(obj_path)
            #code.interact(local=dict(globals(),**locals()))
            i += 1

print(class_dict)

import shutil
from cgan import *

cnt=1
temp=0
path_os = os.getcwd()
source_path=path_os+"/full_forg/"
for images in os.listdir(source_path):
        if images[0]!='f': #check only for forg images
            continue
        img_path = source_path+'/'+images
        if os.path.exists(path_os+'/box'):
            shutil.rmtree(path_os+'/box')
        if os.path.exists(path_os + '/CGAN CEDAR forg/generated_' + images[:-4] + '_1.png'):
            cnt += 1
            print(cnt, end=' ')
            continue
        os.mkdir(path_os+'/box') #temporary input folder for cgan
        os.mkdir(path_os+'/box/img')
        box_path=path_os+'/box/img'
        shutil.copy(img_path,box_path)
        shutil.copy2(box_path+'/'+images, box_path+'/'+str(ip)+'_'+images)#since CGAN does not take only 1 image as input, copy it to make 2 similar images
        kt=1
        generateimg(images)
        shutil.rmtree(path_os+'/box')

import shutil
import os
from vae import *

path_os = os.getcwd()
source_path=path_os+"/full_forg/"
cnt=1
temp=0
for images in os.listdir(source_path):
        if images[0]!='f': #check only for forg images
            continue
        img_path = source_path + '/' + images
        if os.path.exists(path_os+'/box'):
            shutil.rmtree(path_os+'/box')
        if os.path.exists(path_os + '/CGAN CEDAR forg/generated_' + images[:-4] + '_1.png'):
            continue
        os.mkdir(path_os+'/box')
        os.mkdir(path_os+'/box/img')
        box_path=path_os+'/box/img'
        shutil.copy(img_path,box_path)
        generateimg(images)
        shutil.rmtree(path_os+'/box')

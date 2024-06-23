import numpy as np
import os, cv2
from utils import *

path_os = os.getcwd()

full_forg = []
full_real = []
generated = []
ssim=0 #calculate the sum of ssim of the generated images
# read and preprocess the images
for i in range(1, 56):
    for j in range(1, 25):
        # for human images
        s = str(i)
        s1 = str(j)
        forg = preprocess_img(path_os + 'full_forg/forgeries_' + s + '_' + s1 + '.png')
        full_forg.append(ref_img)
        real = preprocess_img(path_os + 'full_org/original_' + s + '_' + s1 + '.png')
        full_real.append(preprocess_img(path_os + 'full_org/original_' + s + '_' + s1 + '.png'))
        for k in range(9):  # for the generated images
            idx=str(k)
            generated_path=path_os+'cgan_CEDAR_forg/generated_forgeries_' + s + '_' + s1 + '_' + idx+'.png'
            ssim+=ssim_cal(generated_path,forg) #calculate the ssim of the generated image with the forgery image
            generated.append(preprocess_img(generated_path))
#save the processed images
with open(path_os + '/CEDAR_forg.npy', 'wb') as f:
    np.save(f, full_forg, )
with open(path_os + '/CEDAR_real.npy', 'wb') as f:
    np.save(f, full_real, )
with open(path_os + '/cgan_CEDAR_forgl.npy', 'wb') as f:
    np.save(f, full_real, )
print("Average SSIM score: "+str(ssim/(55*24*9))) #print the average SSIM score







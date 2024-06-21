import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import numpy as np
from PIL import Image
import os,shutil
import cv2,copy
from skimage.metrics import structural_similarity as ssim

def preprocess_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype("uint8")
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tp = cv2.resize(img, (224, 224,), interpolation=cv2.INTER_CUBIC)
    return tp

def ssim_cal(img_path,ref_img):
    img=cv2.imread(img_path)
    img = preprocess_img(img)
    return ssim(ref_img, img, data_range=img.max() - img.min())

path_os = os.getcwd()

def generateimg(name):
    img=Image.open(path_os +'/box/img/'+name)
    path=path_os +'/box/img/'+name
    height, width = img.size
    if height>512 or width>512:
        height=512
        width=512
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Resize((512, 512)),
        ])
    else:
        transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                ])
    device = "cuda"
    dataset = ImageFolder(path_os +'/box/', transform=transform)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    to_image = transforms.ToPILImage()
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    img = cv2.imread(path_os + '/box/img/' + name)
    img = preprocess_img(img)
    def sample_image():
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        # Sample noise
        z = torch.randn(1, 2).to(device)
        sample = vae.decoder(z).to(device)
        save_image(sample.view(1, 1, width, height),
                   path_os + "/VAE Cedar SSI/generated" + '_' + name[:-4] + ".png")

    def checker(i):
        sample_image(1)
        tp = ssim_cal(path_os + "/VAE Cedar SSI/generated" + '_' + name[:-4] + ".png", img) #sample an image
        if tp <= 0.2 and tp>=0.03: #thresholding
            return 1 #1 means met the thresholding critera
        return 0



    class VAE(nn.Module):
        def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
            super(VAE, self).__init__()

            # encoder part
            self.fc1 = nn.Linear(x_dim, h_dim1)
            self.fc2 = nn.Linear(h_dim1, h_dim2)
            self.fc31 = nn.Linear(h_dim2, z_dim)
            self.fc32 = nn.Linear(h_dim2, z_dim)
            # decoder part
            self.fc4 = nn.Linear(z_dim, h_dim2)
            self.fc5 = nn.Linear(h_dim2, h_dim1)
            self.fc6 = nn.Linear(h_dim1, x_dim)

        def encoder(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            return self.fc31(h), self.fc32(h)  # mu, log_var

        def sampling(self, mu, log_var):
            std = torch.exp(0.5 * log_var).cuda()
            eps = torch.randn_like(std).cuda()
            return eps.mul(std).add_(mu) # return z sample

        def decoder(self, z):
            h = F.relu(self.fc4(z))
            h = F.relu(self.fc5(h))
            return F.sigmoid(self.fc6(h))

        def forward(self, x):
            mu, log_var = self.encoder(x.view(-1, width * height).cuda())
            z = self.sampling(mu, log_var)
            return self.decoder(z), mu, log_var


    # build model
    vae = VAE(x_dim=width * height, h_dim1=512, h_dim2=256, z_dim=2)
    vae.to(device)
    optimizer = optim.Adam(vae.parameters(),lr=0.0005)
    # return reconstruction error + KL divergence losses
    def loss_function(recon_x, x, mu, log_var):
        recon_x, x, mu, log_var = recon_x.cpu(), x.cpu(), mu.cpu(), log_var.cpu()
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, width*height), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD


    def train():
        vae.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, log_var = vae(data)
            loss = loss_function(recon_batch, data, mu, log_var)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

    def test():
        vae.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data.to(device)
                recon, mu, log_var = vae(data)

                # sum up batch loss
                test_loss += loss_function(recon, data, mu, log_var).item()

        test_loss /= len(test_loader.dataset)

    for epoch in range(200):
        train(epoch)
        test()
        if (epoch+1)%4==0: #check each 4 epochs
            tp=checker(epoch+1)
            if tp==1:
                break #stop when meet the criteria

cnt=1
temp=0
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
        if os.path.exists(path_os + '/CGAN CEDAR forg/generated_' + images[:-4] + '.png'):
            continue
        os.mkdir(path_os+'/box')
        os.mkdir(path_os+'/box/img')
        box_path=path_os+'/box/img'
        shutil.copy(img_path,box_path)
        generateimg(images)
        shutil.rmtree(path_os+'/box')

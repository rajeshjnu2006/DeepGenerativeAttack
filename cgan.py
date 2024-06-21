import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader

import numpy as np
from PIL import Image
import os
import easydict


def generateimg(name):
    opt = easydict.EasyDict({
        "batch_size": 9,
        "n_epochs": 800,
        "n_cpu": 8,
        "latent_dim": 20,
        "n_classes": 2,
        "img_size1": 512,
        "img_size2": 512,
        "channels": 1,
        "lr": 0.001,
        "b1": 0.5,
        "b2": 0.999,
        "sample_interval": 50,
    })


    path_os = os.getcwd()
    img = Image.open(path_os + '/box/img/' + name) #input image path
    path = path_os + '/box/img/' + name
    height, width = img.size
    if height < 512 and width < 512: #resizing image to avoid memory exceeding
        opt.img_size2= height
        opt.img_size1=width
    img_shape = (opt.channels, opt.img_size1, opt.img_size2)
    transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Resize((opt.img_size1, opt.img_size2)),
        ])
    device = "cuda"
    #device = 'cpu'
    cuda = True #if torch.cuda.is_available() else False

    to_image = transforms.ToPILImage()
    dataset = datasets.ImageFolder(path_os+'/box', transform=transform) #read the image
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    def sample_image(): #save a sample of image
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        # Sample noise
        z = Variable(FloatTensor(np.random.normal(0, 1, (opt.n_classes, opt.latent_dim))))
        labels = np.array([num for num in range(opt.n_classes)])
        labels = Variable(LongTensor(labels))
        gen_imgs = generator(z, labels)
        img = gen_imgs.data
        for i in range(opt.n_classes):
            cut = len(img) // opt.n_classes
            save_img = img[i * cut:(i + 1) * cut]
            save_image(save_img, path_os+"/CGAN CEDAR forg/generated"+'_'+name[:-4]+'_'+ str(i+1)+ ".png",
                       normalize=True)

    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()

            self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

            def block(in_feat, out_feat, normalize=True):
                layers = [nn.Linear(in_feat, out_feat)]
                if normalize:
                    layers.append(nn.BatchNorm1d(out_feat, 0.8))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return layers

            self.model = nn.Sequential(
                *block(opt.latent_dim + opt.n_classes, 128, normalize=False),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, int(np.prod(img_shape))),
                nn.Tanh()
            )

        def forward(self, noise, labels):
            # Concatenate label embedding and image to produce input
            gen_input = torch.cat((self.label_emb(labels), noise), -1)
            img = self.model(gen_input)
            img = img.view(img.size(0), *img_shape)
            return img

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()

            self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

            self.model = nn.Sequential(
                nn.Linear(opt.n_classes + int(np.prod(img_shape)), 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 512),
                nn.Dropout(0.4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 512),
                nn.Dropout(0.4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 1),
            )

        def forward(self, img, labels):
            # Concatenate label embedding and image to produce input
            d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
            validity = self.model(d_in)
            return validity

    # Loss functions
    adversarial_loss = torch.nn.MSELoss().to(device)

    # Initialize generator and discriminator
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


    # build model
    for epoch in range(opt.n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            batches_done = epoch * len(dataloader) + i + 1
            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            labels = torch.tensor([num for num in range(opt.n_classes)])
            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)
            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2


            d_loss.backward()
            optimizer_D.step()

            if batches_done % opt.sample_interval == 0:

                sample_image(batches_done=batches_done)
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )







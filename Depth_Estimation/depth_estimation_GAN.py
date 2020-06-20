import os
import random
import cv2
import load_data
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # (1) Encoder
            # input is 3 x 150 x 150
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 75 x 75
            nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 38 x 38
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 19 x 19
            nn.Conv2d(ndf * 4, ndf * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 10 x 10

            nn.Conv2d(ndf * 8, ndf*16, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 7 x 7

            nn.Conv2d(ndf*16, ndf*32, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*32) x 4 x 4

            nn.Conv2d(ndf*32, ndf*64, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            # state size (ndf*64) x 1 x 1

            # (2) Decoder
            # input is Z, going into a convolution
            nn.ConvTranspose2d(ndf*64, ngf*16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*16),
            nn.ReLU(True),
            # Print(),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf*16, ngf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            # Print(),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 0, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # Print(),
            # state size. (ngf*4) x 18 x 18
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 0, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # Print(),
            # state size. (ngf*2) x 37 x 37
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # Print(),
            # state size. (ngf) x 74 x 74
            nn.ConvTranspose2d(ngf, 1, 4, 2, 0, bias=False),
            nn.Tanh(),
            # Print(),
            # state size. (1) x 150 x 150
        )

    def forward(self, input):
        return self.main(input)


# class Discriminator(nn.Module):
#     def __init__(self, ngpu):
#         super(Discriminator, self).__init__()
#         self.ngpu = ngpu
#         self.main = nn.Sequential(
#             # input is (nc) x 64 x 64
#             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 32 x 32
#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 16 x 16
#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*4) x 8 x 8
#             nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*8) x 4 x 4
#             nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, input):
#         return self.main(input)
#

class Print(nn.Module):
    def forward(self, x):
        print("The size is:", x.size())
        return x


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 150 x 150
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 75 x 75
            nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 38 x 38
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 19 x 19
            nn.Conv2d(ndf * 4, ndf * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 10 x 10

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 1 x 7 x 7

            nn.Conv2d(1, 1, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 1 x 4 x 4

            nn.Conv2d(1, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # state size 1 x 1 x 1

            # # state flattened. 49 input & 32 output
            # # Print(),
            # Flatten(),
            # nn.Linear(ndf*8*10*10, 1024),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. 32 input & 2 output
            # nn.Linear(1024, 1),
        )

    def forward(self, input):
        return self.main(input)


def training():
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    generated_images = []
    print("Starting training loop")

    for epoch in range(num_epochs):
        print("Epoch", epoch+1, "of", num_epochs)
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with all-real batch
            netD.zero_grad()
            # Format batch
            # print(np.shape(data))
            real_cpu = data.to(device)
            # print(np.shape(real_cpu))
            # real_cpu = real_cpu.reshape(1, real_cpu.size(0), real_cpu.size(1), real_cpu.size(2))
            b_size = real_cpu.size(0)
            # print(b_size)

            # label = []
            # for num in range(len(data)):
            #     label.append([real_label, 0])
            # label = torch.tensor(label, dtype=torch.float32)
            # label = label.to(device)

            label = torch.full((len(data),), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # print(np.shape())
            # Calculate loss on all-real batch
            # print("Shapes:")
            # print(np.shape(output))
            # print(np.shape(label))
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            rgb_images = [x[0:3] for x in data]
            rgb_images = torch.stack(rgb_images)
            # rgb_images = torch.tensor(rgb_images, dtype=torch.float32)
            rgb_images = rgb_images.to(device)
            # print(rgb_images.size())
            # print(len(rgb_images[0]))
            # print(len(rgb_images[0][0]))
            # print(len(rgb_images[0][0][0]))
            # Generate fake image batch with G
            # print(np.shape(data[:, [], :, :]))
            fake = netG(rgb_images)
            if i == 0:
                # print("Appending")
                generated_images.append(fake[i])
                # print("Appended")
            label.fill_(fake_label)

            # label = []
            # for num in range(len(data)):
            #     label.append([0, fake_label])
            # label = torch.tensor(label, dtype=torch.float32)
            # label = label.to(device)

            # print("The dimension of fake is:", np.shape(fake))
            # print("The shape of data is:", np.shape(data[0]), np.shape(data[0][0]))
            fake_combined = data.to(device)
            fake_combined[:, [3], :, :] = fake.to(device)
            # for j in range(batch_size):
            #     if j == 0:
            #         print(np.shape(torch.stack((data[j][0].to(device), data[j][1].to(device), data[j][2].to(device), fake[j][0].to(device)))))
            #     fake_combined.append(torch.stack((data[j][0].to(device), data[j][1].to(device), data[j][2].to(device), fake[j][0].to(device))))
            fake_combined = torch.tensor(fake_combined, dtype=torch.float32)
            fake_combined = fake_combined.to(device)
            # print(np.shape(data[:][0:3]))
            # print(np.shape(fake))
            # fake_combined = torch.cat((data[:][0:3].to(device), fake.to(device)), dim=1)
            # print("The dimension of fake combined is:", np.shape(fake_combined))
            # print("The type of fake combines is:", type(fake))
            # Classify all fake batch with D
            output = netD(fake_combined.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost

            # label = []
            # for num in range(len(data)):
            #     label.append([real_label, 0])
            # label = torch.tensor(label, dtype=torch.float32)
            # label = label.to(device)

            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake_combined).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # # Check how the generator is doing by saving G's output on fixed_noise
            # if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            #     with torch.no_grad():
            #         fake = netG(fixed_noise).detach().cpu()
            #     img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Fake Images")
    # print(np.shape(generated_images))
    # print(np.shape(generated_images[0]))
    for i in range(len(generated_images)):
        temp = generated_images[i].cpu().detach().numpy().swapaxes(0, 2).swapaxes(0, 1)
        cv2.imshow('Generated Image', temp)
        cv2.waitKey(0)
    # plt.imshow(np.transpose(generated_images[0].cpu().detach().numpy().swapaxes(0, 2), (1, 2, 0)))
    # plt.show()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    dataroot = r""
    workers = 2
    batch_size = 128
    image_size = 150
    nc = 4
    nz = 100
    ngf = 64
    ndf = 64
    num_epochs = 1
    lr_d = 0.0003
    lr_g = 0.003
    beta1 = 0.5
    ngpu = 1

    dataset = load_data.load_data()
    transform = transforms.Compose(
        [
            transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5])
        ]
    )
    dataset = torch.tensor(dataset, dtype=torch.float32)
    # dataset = transform(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    print(netG)

    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)
    print(netD)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    real_label = 0
    fake_label = 1

    optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, 0.999))

    print("Begin Training")
    training()

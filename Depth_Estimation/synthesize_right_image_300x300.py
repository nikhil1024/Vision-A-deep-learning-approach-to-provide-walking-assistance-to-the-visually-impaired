import torch
from torch import nn
from torch import optim
import torch.utils.data
import torch.nn.functional as F
from torchvision import transforms
import torchvision as tv
import cv2
from tqdm import tqdm
from random import shuffle
import os
import numpy as np
import matplotlib.pyplot as plt


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # state size. 1 x 300 x 300
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.deconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=33, kernel_size=4, stride=2, padding=0)

        # state size. 32 x 149 x 149
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=33, kernel_size=12, stride=4, padding=0)

        # state size. 64 x 73 x 73
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=33, kernel_size=28, stride=8, padding=0)

        # state size. 128 x 35 x 35
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.bn4 = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(in_channels=256, out_channels=33, kernel_size=15, stride=19, padding=0)

        # state size. 256 x 16 x 16
        self.linear1 = nn.Linear(256 * 16 * 16, 4096)
        self.linear2 = nn.Linear(4096, 2700)

        self.deconv5 = nn.ConvTranspose2d(in_channels=3, out_channels=33, kernel_size=10, stride=10, padding=0)

        self.conv5 = nn.Conv2d(in_channels=33, out_channels=33, kernel_size=5, stride=1, padding=2)

        self.conv6 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        deconv = []
        # print(x.size())
        # print(type(x))
        # transform = transforms.Compose([
        #                                 transforms.ToPILImage(),
        #                                 transforms.Grayscale(num_output_channels=1),
        #                                 transforms.ToTensor()
        #                                 ])
        # # cannot send entire batch to convert to PIL image. Need to send individual images
        # gray_scale = transform(x[0].to('cpu'))
        # gray_scale = gray_scale.view(-1, 1, 150, 150)
        # print(gray_scale.size())
        shifted_images = shift_images(x, -16, 16).to(device)
        # x = x.to(device)
        # x = x.view(-1, 3, 150, 150)

        x = self.bn1(self.pool1(F.elu(self.conv1(x))))
        # print(x.size())
        deconv.append(self.deconv1(x))
        # print(self.deconv1(x).size())

        x = self.bn2(self.pool2(F.elu(self.conv2(x))))
        # print(x.size())
        deconv.append(self.deconv2(x))
        # print(self.deconv2(x).size())

        x = self.bn3(self.pool3(F.elu(self.conv3(x))))
        # print(x.size())
        deconv.append(self.deconv3(x))
        # print(self.deconv3(x).size())

        x = self.bn4(self.pool4(F.elu(self.conv4(x))))
        # print(x.size())
        deconv.append(self.deconv4(x))
        # print(self.deconv4(x).size())

        # Flatten()
        x = x.view(-1, 256 * 16 * 16)
        x = F.elu(self.linear1(x))
        x = F.elu(self.linear2(x))

        x = x.view(-1, 3, 30, 30)  # 30 x 30 x 3
        # print(x.size())
        deconv.append(self.deconv5(x))
        # print(self.deconv5(x).size())

        combined_deconv = deconv[0]
        for i in range(1, len(deconv)):
            combined_deconv.add_(deconv[i])

        # print(combined_deconv.size())

        combined_conv = F.elu(self.conv5(combined_deconv))
        # print(combined_conv.size())
        combined_conv = F.elu(self.conv5(combined_conv))
        # print(combined_conv.size())
        # combined_conv = combined_conv.to(device)

        product = torch.mul(combined_conv, shifted_images)
        # print("Product:", product.size())
        selection_layer_output = torch.sum(product, dim=1).view(-1, 1, height, width)
        # print("Selection layer output:", selection_layer_output.size())

        prediction = F.elu(self.conv6(selection_layer_output))
        prediction = F.elu(self.conv6(prediction))
        # print(prediction.size())

        return prediction


def weights_init(m):
    classname = m.__class__.__name__
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def shift_images(batch, start, end):
    new_batch = []
    for n1, img in enumerate(batch):
        n = 0
        combined_img = []
        for r in range(start, end + 1):
            if r > 0:
                combined_img.append([[0 for _ in range(r)] + y[:-r].tolist() for y in img[n]])
            elif r < 0:
                combined_img.append([y[-r:].tolist() + [0 for _ in range(-r)] for y in img[n]])
            else:
                combined_img.append([y.tolist() for x in img for y in x])
        new_batch.append(combined_img)
    new_batch = torch.tensor(new_batch, dtype=torch.float32)
    return new_batch


def training():
    loss = []

    for epoch in range(num_epochs):
        print("Epoch", epoch + 1, "of", num_epochs)
        iters = 0
        for data, label in zip(dataloader1, dataloader2):
            model.zero_grad()
            data = data.to(device)
            label = label.to(device)
            prediction = model(data)
            err = criterion(prediction, label)
            err.backward()
            optimizer.step()
            iters += 1

        print("Loss: %0.4f", err.item())
        loss.append(err.item())

    model_save_name = 'model_{}_{}_300x300.pt'.format(num_epochs, lr)
    model_save_path = r"drive/My Drive/B. Tech Project/Trained Models/Depth Estimation/{}".format(model_save_name)
    # model_save_path = r"drive/My Drive/B. Tech Project/Trained Models/Depth Estimation/test.pt"
    torch.save(model.state_dict(), model_save_path)


def testing():
    model_load_path = r'D:\model_50_0.0003_300x300_elu.pt'
    model.load_state_dict(torch.load(model_load_path))
    model.eval()
    left_image_path = r"D:\left.png"
    left_image = cv2.resize(cv2.imread(left_image_path, cv2.CV_8UC1), (300, 300))
    blur = cv2.GaussianBlur(left_image, (7, 7), 0)
    filtered = cv2.subtract(left_image, blur)
    blurred_left_image = cv2.GaussianBlur(left_image, (11, 11), 0)

    kernel = np.array([[0, 0, -1, 0, 0],
                       [0, 0, -1, 0, 0],
                       [-1, -1, 9, -1, -1],
                       [0, 0, -1, 0, 0],
                       [0, 0, -1, 0, 0]])
    # left_image = cv2.filter2D(left_image, -1, kernel)

    cv2.imshow('Sharpened Left Image', left_image)
    cv2.waitKey(0)
    cv2.imshow('Sharpened Image', filtered)
    cv2.waitKey(0)
    transform_toTensor = transforms.Compose(
        [
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]
    )
    # print(type(left_image))
    left_image = torch.tensor(left_image, dtype=torch.float32)
    # norm = Normalize(mean=(0.5), std=(0.5))
    # left_image = norm(left_image)
    left_image = left_image.view(1, 300, 300)
    # left_image = transform_toTensor(left_image)
    # cv2.imshow('Original right image', np.array(train_y[0].view(150, 150).cpu().detach().numpy(), dtype=np.uint8))
    # cv2.waitKey(0)
    # generated_right_image = model(train_x[0].view(-1, 1, 150, 150).to(device)).to(device).view(1, 150, 150)
    generated_right_image = model(left_image.view(-1, 1, 300, 300).to(device)).to(device).view(1, 300, 300)
    inv_normalize = transforms.Compose(
        [
            transforms.Normalize(mean=[-0.5/0.5], std=[1/0.5])
        ]
    )
    # unnorm = UnNormalize(mean=(0.5), std=(0.5))
    # generated_right_image = unnorm(generated_right_image)
    print(generated_right_image.size())
    # generated_right_image = inv_normalize(generated_right_image)
    transform_toImage = transforms.Compose(
        [
            transforms.ToPILImage()
        ]
    )
    # generated_right_image = transform_toImage(generated_right_image)
    generated_right_image = generated_right_image.view(300, 300).cpu().detach().numpy()
    print(generated_right_image)
    max_val = np.max(generated_right_image)
    print(max_val)
    decreasing_factor = 255/max_val
    temp = generated_right_image
    temp2 = generated_right_image
    for i in range(len(generated_right_image)):
        for j in range(len(generated_right_image[i])):
            generated_right_image[i][j] *= decreasing_factor
            # if generated_right_image[i][j] > 255:
            #     generated_right_image[i][j] = 255
            # if generated_right_image[i][j] > 500:
            #     generated_right_image[i][j] = 255
            # elif generated_right_image[i][j] > 300:
            #     generated_right_image[i][j] = 225
            # elif generated_right_image[i][j] > 200:
            #     generated_right_image[i][j] = 200
            # elif generated_right_image[i][j] > 150:
            #     generated_right_image[i][j] = 175
            # elif generated_right_image[i][j] > 100:
            #     generated_right_image[i][j] = 125
            # elif generated_right_image[i][j] > 50:
            #     generated_right_image[i][j] = 50
            # else:
            #     generated_right_image[i][j] = 0
    # for i in range(len(generated_right_image)):
    #     for j in range(len(generated_right_image[i])):
    #         if temp2[i][j] > 255:
    #             temp2[i][j] = 255
    generated_right_image = generated_right_image.astype(int)
    # temp = temp.astype(int)
    # temp2 = temp2.astype(int)
    print(generated_right_image)
    print(np.max(generated_right_image))
    generated_right_image = np.array(generated_right_image, dtype=np.uint8)
    # temp = np.array(temp, dtype=np.uint8)
    # temp2 = np.array(temp2, dtype=np.uint8)
    # blur = cv2.GaussianBlur(generated_right_image, (7, 7), 0)
    # filtered = cv2.subtract(generated_right_image, blur)

    kernel = np.array([[0, 0, -1, 0, 0],
                       [0, 0, -1, 0, 0],
                       [-1, -1, 9, -1, -1],
                       [0, 0, -1, 0, 0],
                       [0, 0, -1, 0, 0]])
    sharpened = cv2.filter2D(generated_right_image, -1, kernel)

    cv2.imshow('Generated Image', np.array(generated_right_image, dtype=np.uint8))
    cv2.waitKey(0)
    cv2.imwrite(r'C:\Users\Nikhil\Desktop\generated_right_image.jpg', np.array(generated_right_image, dtype=np.uint8))
    # cv2.imshow('Generated Right Image 1', np.array(temp, dtype=np.uint8))
    # cv2.waitKey(0)
    # cv2.imshow('Generated Right Image 2', np.array(temp2, dtype=np.uint8))
    # cv2.waitKey(0)
    cv2.imshow('Sharpened Image', np.array(sharpened, dtype=np.uint8))
    cv2.waitKey(0)

    print(left_image.shape)
    print(generated_right_image.shape)
    left_image = left_image.view(300, 300)
    left_image = np.array(left_image, dtype=np.uint8)
    # generated_right_image = np.array(generated_right_image, dtype=np.uint8)
    print(generated_right_image.dtype)
    # generated_right_image = cv2.cvtColor(generated_right_image, cv2.CV_8UC1)
    stereo = cv2.StereoBM_create(numDisparities=32, blockSize=11)
    disparity = stereo.compute(blurred_left_image, generated_right_image)
    cv2.imshow('Disparity', np.array(disparity, dtype=np.uint8))
    cv2.waitKey(0)
    plt.imshow(disparity)
    plt.show()

    _, axs = plt.subplots(1, 3, figsize=(12, 12))
    axs = axs.flatten()
    for img, ax in zip([left_image, sharpened, disparity], axs):
        ax.imshow(img)
    plt.show()


def get_data():
    dataset_dir = r"F:\Datasets\KITTI drive"
    data = []

    for root_dir in tqdm(os.listdir(dataset_dir)):
        path = os.path.join(dataset_dir, root_dir)
        for sub_dir in os.listdir(path):
            count = 0
            count_left, count_right = 0000000000, 0000000000
            path = os.path.join(path, sub_dir)
            left_image_path = os.path.join(path, "image_02", "data")
            right_image_path = os.path.join(path, "image_03", "data")

            for _ in range(len(os.listdir(left_image_path))):
                if count % 2 == 0:
                    left_image = cv2.resize(cv2.imread(os.path.join(left_image_path, str(count_left).zfill(10) + ".png"), cv2.IMREAD_GRAYSCALE), (300, 300))
                    count_left += 1
                else:
                    right_image = cv2.resize(cv2.imread(os.path.join(right_image_path, str(count_right).zfill(10) + ".png"), cv2.IMREAD_GRAYSCALE), (300, 300))
                    count_right += 1

                    data.append([np.array(left_image), np.array(right_image)])

                count += 1

    shuffle(data)

    train_set = data[:-int(len(data) * 0.3)]
    test_set = data[-int(len(data) * 0.3):]
    train_x, train_y, test_x, test_y = [], [], [], []

    for sample in range(len(train_set)):
        train_x.append(train_set[sample][0])
        train_y.append(train_set[sample][1])

    for sample in range(len(test_set)):
        test_x.append(test_set[sample][0])
        test_y.append(test_set[sample][1])

    return train_x, train_y, test_x, test_y


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

lr = 0.0003
weight_decay = 1e-6
num_epochs = 50
batch_size = 16
start = -16
end = 16
height = 300
width = 300
model = ConvNet().to(device)


if __name__ == '__main__':
    # image = cv2.resize(cv2.imread(r"F:\Datasets\KITTI drive\2011_09_26_0093\2011_09_26_drive_0093_sync\image_02\data\0000000000.png"), (150, 150))
    # image = torch.tensor(image, dtype=torch.float32)
    # transform = transforms.Compose([transforms.ToTensor()])
    train_x, train_y, test_x, test_y = get_data()

    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    test_x = torch.tensor(test_x, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)

    transform = transforms.Compose(
        [
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]
    )
    #     train_x = transform(train_x)
    #     train_y = transform(train_y)
    #     test_x = transform(test_x)
    #     test_y = transform(test_y)

    train_x = train_x.view(-1, 1, height, width)
    train_y = train_y.view(-1, 1, height, width)
    test_x = test_x.view(-1, 1, height, width)
    test_y = test_y.view(-1, 1, height, width)

    print("The shape of train x is:", train_x.size())
    print("The shape of train y is:", train_y.size())
    print("The shape of test x is:", test_x.size())
    print("The shape of test y is:", test_y.size())

    # image = transform(image)
    # print(image.size())
    # image = image.view(1, 3, 150, 150).to(device)
    dataloader1 = torch.utils.data.DataLoader(train_x, batch_size=batch_size, shuffle=False)
    dataloader2 = torch.utils.data.DataLoader(train_y, batch_size=batch_size, shuffle=False)

    # model.apply(weights_init)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # output = model(image)
    # training()
    testing()

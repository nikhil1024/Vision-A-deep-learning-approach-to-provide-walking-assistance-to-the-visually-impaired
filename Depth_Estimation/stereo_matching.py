import torch
from torch import nn
from torch import optim
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Function
from torchvision import transforms
import cv2
import numpy as np
import os
from random import shuffle
import matplotlib.pyplot as plt
from tqdm import tqdm
import synthesize_right_image_300x300
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
import spatial_correlation_sampler_backend as correlation


class StereoMatcher(nn.Module):
    def __init__(self):
        super(StereoMatcher, self).__init__()
        # state size. 300 x 300 x 64
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)

        # state size. 300 x 300 x 128
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.bn3 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=1, kernel_size=2, stride=2, padding=0)

        # state size. 150 x 150 x 256
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.bn4 = nn.BatchNorm2d(512)
        self.deconv4 = nn.ConvTranspose2d(in_channels=512, out_channels=1, kernel_size=4, stride=4, padding=0)

        # state size. 75 x 75 x 512
        self.conv5 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2)
        self.conv6 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2)

    def forward(self, left, right):
        deconv = []

        left = F.elu(self.conv1(left))
        left = F.elu(self.conv2(left))

        right = F.elu(self.conv1(right))
        right = F.elu(self.conv2(right))

        corr = spatial_correlation_sample(input1=left, input2=right).view(-1, 1, 300, 300)
        corr = F.elu(self.conv1(corr))
        corr = F.elu(self.conv2(corr))

        concat = torch.cat([corr, right], dim=1)

        concat = self.bn3(self.pool3(F.elu(self.conv3(concat))))
        deconv.append(self.deconv3(concat))

        concat = self.bn4(self.pool4(F.elu(self.conv4(concat))))
        deconv.append(self.deconv4(concat))

        combined_deconv = deconv[0]
        for i in range(1, len(deconv)):
            combined_deconv.add_(deconv[i])

        prediction = F.elu(self.conv5(combined_deconv))
        prediction = F.elu(self.conv6(prediction))

        return prediction


class SpatialCorrelationSamplerFunction(Function):
    def __init__(self,
                 kernel_size,
                 patch_size,
                 stride,
                 padding,
                 dilation_patch):
        super(SpatialCorrelationSamplerFunction, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.patch_size = _pair(patch_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation_patch = _pair(dilation_patch)

    def forward(self, input1, input2):
        self.save_for_backward(input1, input2)
        kH, kW = self.kernel_size
        patchH, patchW = self.patch_size
        padH, padW = self.padding
        dilation_patchH, dilation_patchW = self.dilation_patch
        dH, dW = self.stride

        output = correlation.forward(input1, input2,
                                     kH, kW, patchH, patchW,
                                     padH, padW, dilation_patchH, dilation_patchW,
                                     dH, dW)

        return output

    @once_differentiable
    def backward(self, grad_output):
        input1, input2 = self.saved_variables

        kH, kW = self.kernel_size
        patchH, patchW = self.patch_size
        padH, padW = self.padding
        dilation_patchH, dilation_patchW = self.dilation_patch
        dH, dW = self.stride

        grad_input1, grad_input2 = correlation.backward(input1, input2, grad_output,
                                                        kH, kW, patchH, patchW,
                                                        padH, padW,
                                                        dilation_patchH, dilation_patchW,
                                                        dH, dW)
        return grad_input1, grad_input2


def spatial_correlation_sample(input1,
                               input2,
                               kernel_size=1,
                               patch_size=1,
                               stride=1,
                               padding=0,
                               dilation_patch=1):
    """Apply spatial correlation sampling on from input1 to input2,
    Every parameter except input1 and input2 can be either single int
    or a pair of int. For more information about Spatial Correlation
    Sampling, see this page.
    https://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15/
    Args:
        input1 : The first parameter.
        input2 : The second parameter.
        kernel_size : total size of your correlation kernel, in pixels
        patch_size : total size of your patch, determining how many
            different shifts will be applied
        stride : stride of the spatial sampler, will modify output
            height and width
        padding : padding applied to input1 and input2 before applying
            the correlation sampling, will modify output height and width
        dilation_patch : step for every shift in patch
    Returns:
        Tensor: Result of correlation sampling
    """
    corr_func = SpatialCorrelationSamplerFunction(kernel_size,
                                                  patch_size,
                                                  stride,
                                                  padding,
                                                  dilation_patch)
    return corr_func(input1, input2)


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


def training():
    loss = []

    model = StereoMatcher().to(device)
    model.apply(weights_init)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        print("Epoch", epoch+1, "of", num_epochs)
        iters = 0
        for left, right, depth in zip(dataloader1, dataloader2, dataloader3):
            model.zero_grad()
            left = left.to(device)
            right = right.to(device)
            depth = depth.to(device)

            prediction = model(left, right)
            err = criterion(prediction, depth)
            err.backward()
            optimizer.step()
            iters += 1

        print("Loss: %0.4f", err.item())
        loss.append(err.item())

        model_save_name = 'stereo_matcher_{}_{}.pt'.format(num_epochs, lr)
        model_save_path = r'F:\PyCharm Projects\Audio Assistance to the Visually Impaired\Depth_Estimation\{}'.format(model_save_name)
        torch.save(model.state_dict(), model_save_path)


def testing():
    model = StereoMatcher().to(device)
    model_load_path = r'D:\stereo_matcher_grayscale_50_0.0003.pt'
    model.load_state_dict(torch.load(model_load_path))
    model.eval()

    # depth_map_path = r"F:\Datasets\KITTI drive\2011_09_26_0001\2011_09_26_drive_0001_sync\depths\0000000001.png"
    # depth_map = cv2.resize(cv2.imread(depth_map_path), (300, 300))
    # cv2.imshow('Depth Map', depth_map)
    # cv2.waitKey(0)

    left_image_path = r"F:\Datasets\KITTI drive\2011_09_26_0096\2011_09_26_drive_0096_sync\image_02\data\0000000000.png"
    # left_image_path = r"D:\left.png"
    left_image = cv2.resize(cv2.imread(left_image_path), (300, 300))
    cv2.imshow('Left Image', left_image)
    cv2.waitKey(0)
    print(left_image)
    left_image = cv2.resize(cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE), (300, 300))
    left_image = torch.tensor(left_image, dtype=torch.float32).view(-1, 1, 300, 300)
    left_image = left_image.to(device)

    generated_right_image_path = r"F:\Datasets\KITTI drive\2011_09_26_0096\2011_09_26_drive_0096_sync\image_03\data\0000000000.png"
    # generated_right_image_path = r"C:\Users\Nikhil\Desktop\generated_right_image.jpg"
    generated_right_image = cv2.resize(cv2.imread(generated_right_image_path, cv2.IMREAD_GRAYSCALE), (300, 300))
    generated_right_image = torch.tensor(generated_right_image, dtype=torch.float32).view(-1, 1, 300, 300)
    generated_right_image = generated_right_image.to(device)

    print()
    print()
    # print(depth_map)
    predicted_depth = model(left_image, generated_right_image)
    print("The predicted depth values are:")
    print(predicted_depth)
    print(predicted_depth.size())
    predicted_depth = predicted_depth.view(300, 300)
    print(predicted_depth.size())
    predicted_depth = predicted_depth.cpu().detach().numpy()
    temp = predicted_depth.astype(int)
    cv2.imshow('Temp', np.array(temp, dtype=np.uint8))
    cv2.waitKey(0)
    max_val = np.max(predicted_depth)
    print(predicted_depth)
    print(max_val)
    decreasing_factor = 255/max_val
    for i in range(len(predicted_depth)):
        for j in range(len(predicted_depth[i])):
            # for k in range(len(predicted_depth[i][j])):
            # predicted_depth[i][j] *= decreasing_factor
            if predicted_depth[i][j] < 50:
                predicted_depth[i][j] += 50
            elif 50 <= predicted_depth[i][j] < 150:
                predicted_depth[i][j] = 150
            elif 150 <= predicted_depth[i][j] < 200:
                predicted_depth[i][j] *= 175/150
            else:
                predicted_depth[i][j] = 255
            # if predicted_depth[i][j] > 255:
            #     predicted_depth[i][j] = 255
    predicted_depth = predicted_depth.astype(int)
    print(predicted_depth)
    print(predicted_depth.shape)
    # predicted_depth = cv2.cvtColor(predicted_depth, cv2.COLORSPACE_BGR)
    cv2.imshow('Predicted Depth', np.array(predicted_depth, dtype=np.uint8))
    cv2.waitKey(0)
    mapped_disp = cv2.convertScaleAbs(predicted_depth, cv2.CV_8UC1, 1, 0)
    cv2.imshow('Mapped Disparity', mapped_disp)
    cv2.waitKey(0)
    colored_disp = cv2.applyColorMap(mapped_disp, cv2.COLORMAP_JET)
    cv2.imshow('Colored Disparity', colored_disp)
    cv2.waitKey(0)
    cv2.imwrite(r"C:\Users\Nikhil\Desktop\predicted_grayscale_disparity.png", mapped_disp)
    cv2.destroyAllWindows()
    # plt.imshow(predicted_depth)
    # plt.show()


def get_generated_images(train_x):
    print("Fetching Generated Right Images")
    model = synthesize_right_image_300x300.ConvNet().to(device)
    model_load_path = r'D:\model_50_0.0003_300x300_elu.pt'
    model.load_state_dict(torch.load(model_load_path))
    model.eval()

    train_x = torch.tensor(train_x, dtype=torch.float32).view(-1, 1, 300, 300)
    train_x = train_x.to(device)
    dataloader = torch.utils.data.DataLoader(train_x, batch_size=1, shuffle=False)
    print(train_x.device)
    # print(model.device)
    right = []
    for i, left in enumerate(dataloader):
        print(i)
        with torch.no_grad():
            temp = model(left).view(-1, 1, 300, 300)
        right.append(temp.to('cpu'))
        del temp
        torch.cuda.empty_cache()

    print(len(right))
    right = right.to(device)
    print(right.size())

    return right


def save_depth_data():
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
            disparity_path = os.path.join(path, "disparity_grayscale_new")

            if not os.path.exists(disparity_path):
                os.makedirs(disparity_path)

            for _ in range(2 * len(os.listdir(left_image_path))):
                if count % 2 == 0:
                    # print(os.path.join(left_image_path, str(count_left).zfill(10) + ".png"))
                    left_image = cv2.resize(cv2.imread(os.path.join(left_image_path, str(count_left).zfill(10) + ".png"), cv2.IMREAD_GRAYSCALE), (1000, 1000))
                    count_left += 1
                else:
                    right_image = cv2.resize(cv2.imread(os.path.join(right_image_path, str(count_right).zfill(10) + ".png"), cv2.IMREAD_GRAYSCALE), (1000, 1000))
                    count_right += 1

                    print(os.path.join(left_image_path, str(count_right-1).zfill(10) + ".png"))
                    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=41)
                    stereo.setMinDisparity(0)
                    disparity = stereo.compute(left_image, right_image)
                    max_val = np.max(disparity)
                    decreasing_factor = 255 / max_val
                    for i in range(len(disparity)):
                        for j in range(len(disparity[i])):
                            disparity[i][j] *= decreasing_factor
                    mapped_disparity = cv2.convertScaleAbs(disparity, cv2.CV_8UC1, 1, 0)
                    cv2.imwrite(os.path.join(disparity_path, str(count_right).zfill(10) + ".png"), mapped_disparity)

                count += 1


def get_data():
    dataset_dir = r"F:\Datasets\KITTI drive"
    data = []

    for root_dir in tqdm(os.listdir(dataset_dir)):
        path = os.path.join(dataset_dir, root_dir)
        for sub_dir in os.listdir(path):
            count = 0
            count_left, count_depth = 0000000000, 0000000000
            path = os.path.join(path, sub_dir)

            left_image_path = os.path.join(path, "image_02", "data")
            depth_path = os.path.join(path, "depths")

            for _ in range(len(os.listdir(left_image_path))):
                if count % 2 == 0:
                    # print(os.path.join(left_image_path, str(count_left).zfill(10) + ".png"))
                    left_image = cv2.resize(cv2.imread(os.path.join(left_image_path, str(count_left).zfill(10) + ".png"), cv2.IMREAD_GRAYSCALE), (300, 300))
                    count_left += 1
                else:
                    depth_image = cv2.resize(cv2.imread(os.path.join(depth_path, str(count_depth+1).zfill(10) + ".png")), (300, 300))
                    count_depth += 1

                    data.append([np.array(left_image), np.array(depth_image)])

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


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    lr = 0.0003
    weight_decay = 1e-6
    num_epochs = 50
    batch_size = 32

    # save_depth_data()

    # train_x, train_y, test_x, test_y = get_data()
    # generated_right_images = get_generated_images(train_x)
    #
    # train_x = torch.tensor(train_x, dtype=torch.float32).view(-1, 1, 300, 300)
    # train_y = torch.tensor(train_y, dtype=torch.float32).view(-1, 3, 300, 300)
    # test_x = torch.tensor(test_x, dtype=torch.float32).view(-1, 1, 300, 300)
    # test_y = torch.tensor(test_y, dtype=torch.float32).view(-1, 3, 300, 300)
    #
    # print(train_x.size())
    # print(train_y.size())
    # print(generated_right_images.size())
    #
    # dataloader1 = torch.utils.data.DataLoader(train_x, batch_size=batch_size, shuffle=False)
    # dataloader2 = torch.utils.data.DataLoader(generated_right_images, batch_size=batch_size, shuffle=False)
    # dataloader3 = torch.utils.data.DataLoader(train_y, batch_size=batch_size, shuffle=False)
    #
    # training()

    testing()

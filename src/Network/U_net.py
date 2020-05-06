import torch
import torch.nn as nn
import torch.optim as opt
import torch.utils.data as ut

from src.Tools.Tools import *

import numpy as np
from src.Data_processing.import_data import *
from src.Data_processing.data_container import *
from src.Data_processing.augment_data import *

#This variable can be used to check if the gpu is being used (if you want to test the program on a laptop without gpu)
gpu_used = False

def init_main_device():
    '''
       Initilize class by loading data and maybe preprocess
       ASSIGN CUDAS IF POSSIBLE
       :return:
    '''

    #Check if gpu is available
    if torch.cuda.is_available():
        device = "cuda:0"
        gpu_used = True
        print("Using GPU")
        check_gpu_card()
    else:
        device = "cpu"
        print("Using CPU")

    #assign gpu to the main_device
    main_device = torch.device(device)

    return main_device

def check_gpu_card():
    '''
    The method tries to check which gpu you are using on youre computer
    :return:
    '''
    try:
        import pycuda.driver as cudas
        print("The device you are using is: ",cudas.Device(0).name())

    except ImportError as e:
        print("Could not find pycuda and thus not show amazing stats about youre GPU, have you installed CUDA?")
        pass

class U_NET(nn.Module):

    def __init__(self):
        '''
        setup
        '''
        super(U_NET, self).__init__()
        self.set_up_network()

    def set_up_network(self):
        '''
        setup the convolutional net
        :return:
        '''

        # U 1
        self.conv1 = Conv(1, 64)
        self.conv2 = Conv(64, 64)

        # U 2
        self.conv3 = Conv(64, 128)
        self.conv4 = Conv(128, 128)

        # U 3
        self.conv5 = Conv(128, 256)
        self.conv6 = Conv(256, 256)

        # U 4
        self.conv7 = Conv(256, 512)
        self.conv8 = Conv(512, 512)

        # U 5 Lowest layer
        self.conv9 = Conv(512, 1024)
        self.conv10 = Conv(1024, 1024)

        # U6
        self.conv11 = Conv(1024, 512)

        # U7
        self.conv12 = Conv(512, 256)

        # U8
        self.conv13 = Conv(256, 128)

        # U9
        self.conv14 = Conv(128, 64)

        # poolings
        self.pool1 = nn.MaxPool2d(2,2)

        # upsampling
        self.up1 = Up_conv(1024, 512)
        self.up2 = Up_conv(512, 256)
        self.up3 = Up_conv(256, 128)
        self.up4 = Up_conv(128, 64)

        # 1x1 convulution
        self.conv1x1 = nn.Conv2d(64, 2, 1)



    def forward(self, x):
        # U1
        x1 = self.conv1(x)
        x1 = self.conv2(x1)

        #U2
        x2 = self.conv3(self.pool1(x1))
        x2 = self.conv4(x2)

        # U3
        x3 = self.conv5(self.pool1(x2))
        x3 = self.conv6(x3)

        # U4
        x4 = self.conv7(self.pool1(x3))
        x4 = self.conv8(x4)

        # U5 lowest
        x5 = self.conv9(self.pool1(x4))
        x5 = self.conv10(x5)

        #Implement up-pass

        # U6
        x6 = self.conv11(torch.cat([x4, self.up1(x5)], dim=1))
        x6 = self.conv8(x6)

        # U7
        x7 = self.conv12(torch.cat([x3, self.up2(x6)], dim=1))
        x7 = self.conv6(x7)

        # U8
        x8 = self.conv13(torch.cat([x2, self.up3(x7)], dim=1))
        x8 = self.conv4(x8)

        # U9
        x9 = self.conv14(torch.cat([x1, self.up4(x8)], dim=1))
        x9 = self.conv2(x9)

        return self.conv1x1(x9)

class Conv(nn.Module):

    def __init__(self, channels_in, channels_out):
        '''
        init
        '''
        super(Conv, self).__init__()
        self.set_up_network(channels_in, channels_out)

    def set_up_network(self, channels_in, channels_out):
        '''
        setup the convolutional net
        :return:
        '''
        self.module = nn.Sequential()
        self.module.add_module("conv",nn.Conv2d(channels_in, channels_out, 3, padding=1))
        self.module.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        res = self.module(x)
        return res


class Up_conv(nn.Module):

    def __init__(self, channels_in, channels_out):
        '''
        init
        '''
        super(Up_conv, self).__init__()
        self.set_up_network(channels_in, channels_out)

    def set_up_network(self, channels_in, channels_out):
        '''
        setup the convolutional net
        :return:
        '''
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
       # self.conv1 = Conv(channels_in, channels_out)
        self.conv2 = nn.Conv2d(channels_in, channels_out, 1)


    def forward(self, x):
        #self.Up_conv(x)
        #return self.conv2
        return self.conv2(self.up(x))

def train(device, epochs, batch_size):
    '''
    Trains the network, the training loop is inspired by pytorchs tutorial, see
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    '''
    u_net = U_NET()
    u_net.to(device)

    frames = 30 # aka length of dataset

    #Load data

    path_train = 'data/'
    raw_train = create_data(path_train, 'train_v', frames)
    raw_labels = create_data(path_train, 'train_l', frames)
    raw_test = create_data(path_train, 'test_v', frames)

    [X_augmented, Y_augmented] = augment(raw_train, raw_labels, 5)
    np.append(raw_train, X_augmented)
    np.append(raw_labels, Y_augmented)

    raw_train = torch.from_numpy(raw_train)
    raw_labels = torch.from_numpy(raw_labels)
    raw_test = torch.from_numpy(raw_test)

    train, train_labels, val, val_labels = split_to_training_and_validation(raw_train, raw_labels, 0.8)

    batch_train = Custom_dataset(train, train_labels)
    batch_val = Custom_dataset(val, val_labels)

    dataloader_train = ut.DataLoader(batch_train, batch_size=batch_size,shuffle=True)
    dataloader_val = ut.DataLoader(batch_val, batch_size=batch_size, shuffle=True)

    #Initilize evaluation and optimizer, optimizer is set to standard-values, might want to change those
    evaluation = nn.CrossEntropyLoss()
    optimizer = opt.SGD(u_net.parameters(), lr=0.001, momentum=0.99)

    for e in range(epochs):
        loss_stat = 0
        for i in dataloader_train:
            train = i["data"]
            label = i["label"]

            print(train.size())

            #reset gradients
            optimizer.zero_grad()
            train = train.to(device=device, dtype=torch.float32)
            out = u_net(train)

            label = train.to(device=device, dtype=torch.long)
            label = label.squeeze(1)
            loss = evaluation(out, label)
            loss.backward()
            optimizer.step()

            loss_stat += loss.item()
        print(loss_stat)
        loss_stat = 0

if __name__ == '__main__':
    main_device = init_main_device()
    train(main_device, 2, 1)



import src.Data_processing.import_data as im
import torch
import torch.nn as nn
import torch.optim as opt
import numpy as np
import src.Data_processing.augment_data as ad;

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
        self.conv7 = Conv(265, 512)
        self.conv8 = Conv(512, 512)

        # U 5 Lowest layer
        self.conv9 = Conv(512, 1024)
        self.conv10 = Conv(1024, 1024)

        # poolings
        self.pool1 = nn.MaxPool2d(2, 2)

        # upsampling
        self.up1 = Up_conv(1024, 512)
        self.up2 = Up_conv(512, 256)
        self.up3 = Up_conv(256, 128)
        self.up4 = Up_conv(128, 64)

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
        self.module.add_module("conv",nn.Conv2d(channels_in, channels_out, 3))
        self.module.add_module("relu", nn.ReLU())

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
        self.conv1 = Conv(channels_in, channels_out)
        self.conv2 = Conv(channels_out, channels_out)


    def forward(self, x):
        #self.Up_conv(x)
        #return self.conv2
        return self.conv2(self.conv1(self.up(x)))

def train(device, epochs):
    '''
    Trains the network, the training loop is inspired by pytorchs tutorial, see
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    '''
    u_net = U_NET()
    u_net.to(device)

    frames = 30 # aka length of dataset

    #Load data
    path_train = '../../data/'
    train_volume = im.create_data(path_train, 'train_v', frames)
    train_labels = im.create_data(path_train, 'train_l', frames)
    test_volume = im.create_data(path_train, 'test_v', frames)

    ad.augment(train_volume[1])

    # #Initilize evaluation and optimizer, optimizer is set to standard-values, might want to change those
    # evaluation = nn.CrossEntropyLoss()
    # optimizer = opt.SGD(u_net.parameters(), lr=0.001, momentum=0.9)
    #
    # for e in range(epochs):
    #     # Shuffle data
    #     index = np.arange(frames)
    #     np.random.shuffle(index)
    #
    #     for i in range(frames):
    #         train = train_volume[index[i]]
    #         label = train_labels[index[i]]
    #
    #         #reset gradients
    #         optimizer.zero_grad()
    #         out = u_net(train)
    #         loss = evaluation(out, label)
    #         loss.backward()
    #         optimizer.step()
    #
    #         loss_stat = loss.item()

if __name__ == '__main__':
    main_device = init_main_device()
    train(main_device, 10)



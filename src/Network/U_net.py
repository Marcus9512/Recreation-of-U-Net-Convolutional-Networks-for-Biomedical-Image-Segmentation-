import torch
import torch.nn as nn

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



    def forward(self, x):
        x =1

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
        self.Up_conv(x)
        return self.conv2

if __name__ == '__main__':
    main_device = init_main_device()
    u_net = U_NET()
    u_net.to(main_device)



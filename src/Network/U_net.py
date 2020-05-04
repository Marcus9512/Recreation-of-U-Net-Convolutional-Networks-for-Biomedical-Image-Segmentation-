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
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)

        # U 2
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)

        # U 3
        self.conv5 = nn.Conv2d(128, 256, 3)
        self.conv6 = nn.Conv2d(256, 256, 3)

        # U 4
        self.conv7 = nn.Conv2d(265, 512, 3)
        self.conv8 = nn.Conv2d(512, 512, 3)

        # U 5 Lowest layer
        self.conv9 = nn.Conv2d(512, 1024, 3)
        self.conv10 = nn.Conv2d(1024, 1024, 3)

        # poolings
        self.pool1 =  nn.MaxPool2d(2, 2)

        # upsamplings
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


    def forward(self, x):
        x =1

class Conv(nn.Module):

    def __init__(self):
        '''
        init
        '''
        super(Conv, self).__init__()
        self.set_up_network()

    def set_up_network(self):
        '''
        setup the convolutional net
        :return:
        '''

    def forward(self, x):
        x = 1


class Up_conv(nn.Module):

    def __init__(self):
        '''
        init
        '''
        super(Up_conv, self).__init__()
        self.set_up_network()

    def set_up_network(self):
        '''
        setup the convolutional net
        :return:
        '''

    def forward(self, x):
        x = 1

if __name__ == '__main__':
    main_device = init_main_device()
    u_net = U_NET()
    u_net.to(main_device)



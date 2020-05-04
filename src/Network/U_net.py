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

class U_NET():

    def __init__(self, main_device):
        '''
        setup
        '''
        self.main_device = main_device
        self.set_up_network()

    def set_up_network(self):
        '''
        setup the convolutional net
        :return:
        '''

    def pre_process(self):
        '''
        If we need some preprocessing
        :return:
        '''

    def train(self):
        '''
        Train the network, maybe should be done on CUDAS
        :return:
        '''

    def calculate_accuracy(self):
        '''
        If requiered, implement this method (Maybe exist preimplemented in Pytorch?)
        :return:
        '''

if __name__ == '__main__':
    main_device = init_main_device()
    u_net = U_NET(main_device)



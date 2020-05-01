import torch

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

def pre_process():
    '''
    If we need some preprocessing
    :return:
    '''

def set_up_network():
    '''
    setup the convolutional net
    :return:
    '''

def train():
    '''
    Train the network, maybe should be done on CUDAS
    :return:
    '''

def calculate_accuracy():
    '''
    If requiered, implement this method (Maybe exist preimplemented in Pytorch?)
    :return:
    '''

if __name__ == '__main__':
    init_main_device()
    train()
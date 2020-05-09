import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as opt
import torch.utils.data as ut
import torch.utils.tensorboard as tb

from src.Tools.Tools import *
from src.Data_processing.import_data import *
from src.Data_processing.data_container import *
from src.Data_processing.augment_data import *
from os import path
from torch.autograd import Function

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
        self.pool1 = nn.MaxPool2d(2)

        # upsampling
        self.up1 = Up_conv(1024, 512)
        self.up2 = Up_conv(512, 256)
        self.up3 = Up_conv(256, 128)
        self.up4 = Up_conv(128, 64)

        # 1x1 convulution
        self.conv1x1 = nn.Conv2d(64, 1, kernel_size=1)
        #torch.nn.init.normal_(self.conv1x1.weight, 0, np.sqrt(2 / 64))

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
        conv = nn.Conv2d(channels_in, channels_out, 3, padding=1)
        torch.nn.init.normal_(conv.weight, 0, np.sqrt(2/(9 * channels_in)))
        self.module.add_module("conv",conv)
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
        self.up =  nn.ConvTranspose2d(channels_in , channels_in // 2, kernel_size=2, stride=2)
        #self.relu = nn.ReLU(inplace=True)

        #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.conv = nn.Conv2d(channels_in, channels_out, 2)
        #torch.nn.init.normal_(self.conv.weight, 0, np.sqrt(2 /  channels_in))

    def forward(self, x):
        #self.conv2(self.up(x))
        return self.up(x)
      

def load_net(device):
    glob_path = os.path.dirname(os.path.realpath("src"))
    p = os.path.join(glob_path, "saved_nets")
    model = torch.load(p)
    evaluate_model_no_label(device, model)

def evaluate_model_no_label(device, u_net):
    frames = 30
    raw_test = create_data(path_train, 'test_v', frames)
    raw_test = torch.from_numpy(raw_test)

    batch_test = Custom_dataset(raw_test, None)

    dataloader_val = ut.DataLoader(batch_test, batch_size=1, shuffle=True)
    summary = tb.SummaryWriter()

    pos = 0
    u_net.eval()
    for j in dataloader_val:
        test = j["data"]
        test = test.to(device=device, dtype=torch.long)

        with torch.no_grad():
            out = u_net(test)
            summary.add_image('test_res', torchvision.utils.make_grid(out), int(pos))
            summary.add_image('test_in', torchvision.utils.make_grid(test), int(pos))

            pos += 1


class diceloss(nn.Module):
    def init(self):
        super(diceloss, self).init()

    def forward(self, prediction, target):
        # saving for backwards:
        self.prediction = prediction
        self.target = target

        # diceloss:
        smooth = 1.
        iflat = prediction.view(-1)
        tflat = target.view(-1)
        self.intersection = (iflat * tflat).sum()
        self.sum = torch.sum(iflat * iflat) + torch.sum(tflat * tflat)
        return 1 - ((2. * self.intersection + smooth) / (self.sum + smooth))

    def backward(self, grad_out):
        gt = self.target / self.sum
        inter_over_sum = self.intersection / (self.sum * self.sum)
        pred = self.prediction[:, 1] * inter_over_sum
        dD = gt * 2 + self.prediction * -4

        grad_in = torch.cat((dD*-grad_output[0], dD * grad_output[0]), 0)
        return grad_in, None


def train(device, epochs, batch_size):
    '''
    Trains the network, the training loop is inspired by pytorchs tutorial, see
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

    SummaryWriter https://pytorch.org/docs/stable/tensorboard.html
    '''
    u_net = U_NET()
    u_net.to(device)

    frames = 30 # aka length of dataset

    #Load data
    path_train = 'data/'
    raw_train = create_data(path_train, 'train_v', frames)
    raw_labels = create_data(path_train, 'train_l', frames)

    raw_train, raw_labels = augment_and_crop(raw_train, raw_labels, 5)
    
    raw_train = torch.from_numpy(raw_train)
    raw_labels = torch.from_numpy(raw_labels)


    train, train_labels, val, val_labels, test, test_labels = split_to_training_and_validation(raw_train, raw_labels, 0.8, 0.0)

    batch_train = Custom_dataset(train, train_labels)
    batch_val = Custom_dataset(val, val_labels)

    dataloader_train = ut.DataLoader(batch_train, batch_size=batch_size,shuffle=True)
    dataloader_val = ut.DataLoader(batch_val, batch_size=batch_size, shuffle=True)

    len_t = len(dataloader_train)
    len_v = len(dataloader_val)

    #Initilize evaluation and optimizer, optimizer is set to standard-values, might want to change those
    evaluation = diceloss()
    optimizer = opt.SGD(u_net.parameters(), lr=0.001, momentum=0.99)
    scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    summary = tb.SummaryWriter()

    print(len_t, len_v)

    for e in range(epochs):
        print("Epoch: ",e," of ",epochs)
        loss_training = 0

        # Training
        u_net.train()
        pos = 0
        for i in dataloader_train:
            train = i["data"]
            label = i["label"]

            #reset gradients
            optimizer.zero_grad()
            train = train.to(device=device, dtype=torch.float32)
            out = u_net(train)
            #out = torch.sign(out)

            summary.add_image('training_out',torchvision.utils.make_grid(out), int(pos)+ e * len_t)
            summary.add_image('training_in', torchvision.utils.make_grid(train), int(pos) + e * len_t)
            summary.add_image('training_label', torchvision.utils.make_grid(label), int(pos) + e * len_t)

            label = label.to(device=device, dtype=torch.float32)

            loss = evaluation(out, label)
            loss.backward()
            optimizer.step()

            loss_training += loss.item()
            pos += 1

        loss_training /= len_t
        loss_val = 0

        # Validation
        u_net.eval()
        pos = 0
        for j in dataloader_val:
            val = j["data"]
            label_val = j["label"]
            val = val.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                out = u_net(val)
                summary.add_image('val_res', torchvision.utils.make_grid(out) , int(pos) + e * len_v)
                summary.add_image('val_in', torchvision.utils.make_grid(val), int(pos) + e * len_v)
                summary.add_image('val_label', torchvision.utils.make_grid(label_val), int(pos) + e * len_v)

                label_val = label_val.to(device=device, dtype=torch.float32)

                loss = evaluation(out, label)
                loss_val += loss.item()
                pos += 1

        loss_val /= len_v

        print("Training loss: ",loss_training)
        print("Validation loss: ", loss_val)
        summary.add_scalar('Loss/train', loss_training, e)
        summary.add_scalar('Loss/val', loss_val, e)

        scheduler.step(loss_val)
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))

    summary.flush()
    summary.close()

    #Evaluation
    glob_path = os.path.dirname(os.path.realpath("src"))
    p = os.path.join(glob_path,"saved_nets")

    if not path.exists(p):
        print("saved_nets not found, creating the directory")
        try:
            os.mkdir(p)
        except OSError as exc:
            raise
    else:
        print("saved_nets found")

    print("Saving network")
    torch.save(u_net.state_dict(), p+'/save.pt')
if __name__ == '__main__':
    main_device = init_main_device()
    train(main_device, epochs=500, batch_size=1)



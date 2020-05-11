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
# import src.Network.U_net2 as u2
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

    def __init__(self, dropout_prob):
        '''
        setup
        '''
        super(U_NET, self).__init__()
        self.set_up_network(dropout_prob)

    def set_up_network(self, dropout_prob):
        '''
        setup the convolutional net
        :return:
        '''

        # U 1
        self.conv1 = Conv(3, 64)
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
        self.pool1 = nn.MaxPool2d(2, stride=2)

        # upsampling
        self.up1 = Up_conv(1024, 512)
        self.up2 = Up_conv(512, 256)
        self.up3 = Up_conv(256, 128)
        self.up4 = Up_conv(128, 64)

        self.dropout = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(2*dropout_prob)
        self.dropout3 = nn.Dropout(5*dropout_prob)

        # 1x1 convulution
        self.conv1x1 = nn.Conv2d(64, 1, kernel_size=1)
        #self.sigmoid = nn.Sigmoid()
        #self.softmax = nn.Softmax()
        #torch.nn.init.normal_(self.conv1x1.weight, 0, np.sqrt(2 / 64))

    def forward(self, x):
        # U1
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        #x1 = self.dropout(x1)

        #U2
        x2 = self.conv3(self.pool1(x1))
        x2 = self.conv4(x2)
        #x2 = self.dropout(x2)

        # U3
        x3 = self.conv5(self.pool1(x2))
        x3 = self.conv6(x3)
        #x3 = self.dropout2(x3)

        # U4
        x4 = self.conv7(self.pool1(x3))
        x4 = self.conv8(x4)
        #x4 = self.dropout2(x4)

        # U5 lowest
        x5 = self.conv9(self.pool1(x4))
        x5 = self.conv10(x5)
        x5 = self.dropout3(x5)

        #Implement up-pass

        # U6
        x6 = self.conv11(torch.cat([self.up1(x5), x4], dim=1))
        x6 = self.conv8(x6)
        #x6 = self.dropout2(x6)

        # U7
        x7 = self.conv12(torch.cat([ self.up2(x6), x3], dim=1))
        x7 = self.conv6(x7)
        #x7 = self.dropout2(x7)

        # U8
        x8 = self.conv13(torch.cat([self.up3(x7),x2], dim=1))
        x8 = self.conv4(x8)
        #x8 = self.dropout(x8)

        # U9
        x9 = self.conv14(torch.cat([ self.up4(x8), x1], dim=1))
        x9 = self.conv2(x9)
        #x9 = self.dropout(x9)

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
        #torch.nn.init.xavier_normal_(conv.weight)

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
        self.up =  nn.ConvTranspose2d(channels_in , channels_out, kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

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
    #evaluate_model_no_label(device, model)


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
        dD = gt * 2 + pred * -4

        grad_in = torch.cat((dD*-grad_out[0], dD * grad_out[0]), 0)
        return grad_in, None
'''
def load_data():
    frames = 30  # aka length of dataset

    path_train = 'data/'
    raw_train = create_data(path_train, 'train_v', frames)
    raw_labels = create_data(path_train, 'train_l', frames)

    #np.asarray(all_imgs)

    [X_deformed, Y_deformed] = augment_and_crop(raw_train, raw_labels, 5)
    raw_train = np.append(raw_train, X_deformed, axis=0)
    raw_labels = np.append(raw_labels, Y_deformed, axis=0)

    raw_train = torch.from_numpy(raw_train)
    raw_labels = torch.from_numpy(raw_labels)

    def resize(imgs):
        img2 = img.resize((256, 256))
        frame = np.zeros((img2.width, img2.height))

'''
'''
   OLD CODE GRAVE
   
   #frames = 30 # aka length of dataset

    #Load data
    #path_train = 'data/'
    #raw_train = create_data(path_train, 'train_v', frames)
    #raw_labels = create_data(path_train, 'train_l', frames)

    #raw_train, raw_labels = augment_and_crop(raw_train, raw_labels, 5)
    #[X_deformed, Y_deformed] = augment(raw_train, raw_labels, 10)
    #raw_train = np.append(raw_train, X_deformed, axis=0)
    #raw_labels = np.append(raw_labels, Y_deformed, axis=0)
    
    #raw_train = torch.from_numpy(raw_train)
    #raw_labels = torch.from_numpy(raw_labels)


    #train, train_labels, val, val_labels, test, test_labels = split_to_training_and_validation(raw_train, raw_labels, 0.8, 0.0)
'''
def train(device, epochs, batch_size):
    '''
    Trains the network, the training loop is inspired by pytorchs tutorial, see
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

    SummaryWriter https://pytorch.org/docs/stable/tensorboard.html
    '''
    u_net = U_NET(0.1)
    u_net.to(device)


    batch_train = Custom_dataset()

    augment()

    batch_train, batch_val = random_split(batch_train, [150, 30])

    dataloader_train = ut.DataLoader(batch_train, batch_size=batch_size,shuffle=True, pin_memory=True)
    dataloader_val = ut.DataLoader(batch_val, batch_size=batch_size, shuffle=False, pin_memory=True)

    len_t = len(dataloader_train)
    len_v = len(dataloader_val)

    #Initilize evaluation and optimizer, optimizer is set to standard-values, might want to change those
    evaluation = nn.BCEWithLogitsLoss()
    diceloss_eval = diceloss()

    optimizer = opt.SGD(u_net.parameters(), lr=0.001,weight_decay=1e-8, momentum=0.9)
    scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

    summary = tb.SummaryWriter()

    print(len_t, len_v)

    # Code for saving network
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

    #Training loop

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
            #print(out)

            #out = torch.sign(out)

            if pos == len_t-2:
                summary.add_image('training_out',torchvision.utils.make_grid(out), int(pos)+ e * len_t)
                summary.add_image('training_in', torchvision.utils.make_grid(train), int(pos) + e * len_t)
                summary.add_image('training_label', torchvision.utils.make_grid(label), int(pos) + e * len_t)

            label = label.to(device=device, dtype=torch.float32)

            #label = label.squeeze(0)

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
                if pos == len_v - 2:
                    summary.add_image('val_res', torchvision.utils.make_grid(out) , int(pos) + e * len_v)
                    summary.add_image('val_in', torchvision.utils.make_grid(val), int(pos) + e * len_v)
                    summary.add_image('val_label', torchvision.utils.make_grid(label_val), int(pos) + e * len_v)

                label_val = label_val.to(device=device, dtype=torch.float32)

                #out = torch.sigmoid(out)
                #out = (out > 0.5).float()

                loss = evaluation(out, label_val)
                loss_val += loss.item()
                pos += 1

        loss_val /= len_v

        print("Training loss: ",loss_training)
        print("Validation loss: ", loss_val)

        summary.add_scalar('Loss/train', loss_training, e)
        summary.add_scalar('Loss/val', loss_val, e)

        scheduler.step(loss_val)

        torch.save(u_net.state_dict(), p + '/save'+str(e)+'pt')
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))

    summary.flush()
    summary.close()

    print("Saving network")
    torch.save(u_net.state_dict(), p+'/save.pt')

def download_coco():
    glob_path = os.path.dirname(os.path.realpath("src"))
    p = os.path.join(glob_path, "coco_img")
    p2 = os.path.join(glob_path, "ann_file")
    torchvision.datasets.CocoCaptions(p, p2, transform=None, target_transform=None, transforms=None)

if __name__ == '__main__':
    main_device = init_main_device()
    train(main_device, epochs=1000, batch_size=1)



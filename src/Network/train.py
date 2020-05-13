from src.Network.U_net import U_NET

import os
import torch
import glob
import torchvision
import torch.nn as nn
import torch.optim as opt
import torch.utils.data as ut
import torch.utils.tensorboard as tb

from src.Tools.Tools import *
from src.Data_processing.data_container import *
from src.Data_processing.augment_data import *
from PIL import Image

from os import path

# Diceloss added as a module to nn
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


def dice_coef(prediction, target):
    # diceloss:
    smooth = 1.
    iflat = prediction.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    sum = torch.sum(iflat * iflat) + torch.sum(tflat * tflat)
    return ((2. * intersection + smooth) / (sum + smooth))

# Training
def train(device, epochs, batch_size, loss_function="cross_ent", use_schedular=False, learn_rate=.001, learn_decay=1e-8, learn_momentum=.99, per_train = 0.5, per_test = 0.25, per_val = 0.25):
    '''
    Trains the network, the training loop is inspired by pytorchs tutorial, see
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    SummaryWriter https://pytorch.org/docs/stable/tensorboard.html
    '''
    u_net = U_NET(0.1)
    u_net.to(device)

    batch_train = Custom_dataset()
    dataset_length = batch_train.len

    #assert(dataset_length == 30*4+30)
    
    to_train = int(dataset_length*per_train)
    to_test = int(dataset_length*per_test)
    to_val = int(dataset_length*per_val)

    sum = to_train + to_test + to_val
    if sum < dataset_length:
        to_val += dataset_length-sum

    batch_train, batch_val, batch_test = random_split(batch_train, [to_train, to_val, to_test])
    
    dataloader_train = ut.DataLoader(batch_train, batch_size=batch_size,shuffle=True, pin_memory=True)
    dataloader_val = ut.DataLoader(batch_val, batch_size=batch_size, shuffle=True, pin_memory=True)
    dataloader_test = ut.DataLoader(batch_test, batch_size=batch_size, shuffle=True, pin_memory=True)

    len_t = len(dataloader_train)
    len_v = len(dataloader_val)
    len_test = len(dataloader_test)

    file_name = ""

    # Initilize evaluation and optimizer, optimizer is set to standard-values, might want to change those
    if loss_function == "cross_ent":
        evaluation = nn.CrossEntropyLoss()
        file_name = "/cross_ent-"+"lr="+str(learn_rate)+"-lr_weight="+str(learn_decay)+"-lr_moment="+str(learn_momentum)+".pt"
    elif loss_function == "dice":
        evaluation = diceloss()
        file_name = "/dice-"+"lr="+str(learn_rate)+"-lr_weight="+str(learn_decay)+"-lr_moment="+str(learn_momentum)+".pt"
    else:
        evaluation = nn.BCEWithLogitsLoss()
        file_name = "/bce-"+"lr="+str(learn_rate)+"-lr_weight="+str(learn_decay)+"-lr_moment="+str(learn_momentum)+".pt"

    optimizer = opt.SGD(u_net.parameters(), lr=learn_rate, weight_decay=learn_decay, momentum=learn_momentum)

    if use_schedular:
        scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

    summary = tb.SummaryWriter()


    print(len_t, len_v, len_test)

    # Code for saving network
    glob_path = os.path.dirname(os.path.realpath("src"))
    p = os.path.join(glob_path,"saved_nets")


    if not path.exists(p):
        print("saved_nets not found, creating the directory")
        try:
            os.mkdirs(p)
        except OSError as exc:
            raise
    else:
        print("saved_nets found")

    p2 = os.path.join(glob_path, "evaluation_images/"+loss_function + "/lr_" + str(learn_rate) + "/lr_decay_" + str(learn_decay) + "/lr_moment_" + str(learn_momentum))

    if not path.exists(p2):
        try:
            os.makedirs(p2)

        except OSError as exc:
            raise

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
                summary.add_image('training_res',torchvision.utils.make_grid(out), int(pos)+ e * len_t)
                summary.add_image('training_in', torchvision.utils.make_grid(train), int(pos) + e * len_t)
                summary.add_image('training_label', torchvision.utils.make_grid(label), int(pos) + e * len_t)

                #dice_t = dice_coef(out, label)
                #print("Dice loss train ",dice_t)
                #summary.add_scalar('Dice_coef/train', dice_t, int(pos) + e * len_t)

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
                if pos == len_v - 2:
                    summary.add_image('val_res', torchvision.utils.make_grid(out) , int(pos) + e * len_v)
                    summary.add_image('val_in', torchvision.utils.make_grid(val), int(pos) + e * len_v)
                    summary.add_image('val_label', torchvision.utils.make_grid(label_val), int(pos) + e * len_v)

                    #dice_v = dice_coef(out, label_val)
                    #print("Dice loss train ", dice_v)
                    #summary.add_scalar('Dice_coef/val', dice_v, int(pos) + e * len_v)

                label_val = label_val.to(device=device, dtype=torch.float32)

                #out = torch.sigmoid(out)
                #out = (out > 0.5).float()

                loss = evaluation(out, label_val)
                loss_val += loss.item()
                pos += 1

        loss_val /= len_v

        print("Training loss: ",loss_training)
        print("Validation loss: ", loss_val)
        if  loss_function == "dice":
            print("Dice co training: ", -(loss_training-1))
            print("Dice co val: ", -(loss_val-1))
            summary.add_scalar('Dice_co/train', -(loss_training-1), e)
            summary.add_scalar('Dice_co/val', -(loss_val-1), e)

        summary.add_scalar('Loss/train', loss_training, e)
        summary.add_scalar('Loss/val', loss_val, e)

        if use_schedular:
            scheduler.step(loss_val)
        if e % 100 == 0:
            torch.save(u_net.state_dict(), p + '/save_tmp'+str(e)+'pt')
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))

    summary.flush()
    summary.close()

    print("Saving network")

    torch.save(u_net.state_dict(), p+file_name)

    #Evaluation
    print("Evaluation")
    u_net.eval()
    pos = 0
    mse_error = 0
    s = 0
    rand_er = 0
    for j in dataloader_test:
        test = j["data"]
        label_test = j["label"]
        test = test.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            out = u_net(test)

            label_test = label_test.to(device=device, dtype=torch.float32)

            summary.add_image('test_res', torchvision.utils.make_grid(out), int(pos))
            summary.add_image('test_in', torchvision.utils.make_grid(test), int(pos))
            summary.add_image('test_label', torchvision.utils.make_grid(label_test), int(pos))

            out = out.squeeze(0)
            out = out.squeeze(0)
            label_test = label_test.squeeze(0)
            label_test = label_test.squeeze(0)

            out = out.cpu().detach().numpy()
            label_test = label_test.cpu().detach().numpy()

            out = np.where(out > 0.5, 1, 0)

            rand_er += rand_error_2(out, label_test)
            error, s1 = pixel_error(out, label_test)
            print_img(error, s1, out, label_test, "Image "+str(pos),p2)


            mse_error += error
            s += s1
            pos += 1
    print("Mse error: ",mse_error/pos," s: ",s/pos)

    print("Rand error: ",rand_er/pos)

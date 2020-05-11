import os
import numpy as np
from torch.utils.data import *
from PIL import Image

class Custom_dataset(Dataset):
    '''
     Custom dataset class to be able to use our dataset
    '''

    def __init__(self):
        self.glob_path_train = os.path.join(os.path.dirname(os.path.realpath("data/train")), 'train')
        self.glob_path_label = os.path.join(os.path.dirname(os.path.realpath("data/label")), 'label')
        print(self.glob_path_train)
        assert len(os.listdir(self.glob_path_train)) == len(os.listdir(self.glob_path_label))
        self.len = len(os.listdir(self.glob_path_train))

        print(os.listdir(self.glob_path_train))

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        p_train = os.path.join(self.glob_path_train, str(item)+'.jpg')
        p_label = os.path.join(self.glob_path_label, str(item) + '.jpg')

        image = Image.open(p_train)
        label = Image.open(p_label)

        image = np.asarray(image)
        label = np.asarray(label)
        #print(image.shape)
        #print(label.shape)

        image = image.transpose((2, 0, 1))
        label = np.expand_dims(label, 2)
        label = label.transpose((2, 0, 1))

        return {"data":image/255 ,"label":label/255}
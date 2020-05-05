from torch.utils.data import *

class Custom_dataset(Dataset):
    '''
     Custom dataset class to be able to use our dataset
    '''

    def __init__(self, data, labels):
        assert len(data) == len(labels)
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return {"data":self.data[item] ,"label":self.labels[item]}
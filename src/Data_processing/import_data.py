import numpy as np
from matplotlib import pyplot
from PIL import Image   
import torch


def create_data(path, type, n_frames):
    """
    Inputs:
    path - Path to the directory for the data
    type - What kind of data will be created
    n_frames - Number of frames for the Tiff file
    """
    if(type == 'train_v'):
        img = Image.open(path + 'train-volume.tif')
    elif(type == 'train_l'):
        img = Image.open(path + 'train-labels.tif')
    elif(type == 'test_v'):
        img = Image.open(path + 'test-volume.tif')
    else:
        raise Exception('Invalid type {}'.format(type))

    all_imgs = []
    for i in range(n_frames):
        try:
            img.seek(i)
            frame = np.zeros((img.width, img.height))
            for j in range(frame.shape[0]):
                for k in range(frame.shape[1]):
                    frame[j,k] = img.getpixel((j, k)) / 255

            frame = np.expand_dims(frame , 2)
            frame = frame.transpose((2, 0, 1))

    
            all_imgs.append(frame)

        except EOFError:
            # Not enough frames in img
            break

    return np.asarray(all_imgs)


def print_img(all_imgs):
    #if(all_imgs == torch.is_tensor):
    #    all_imgs = all_imgs.eval()


    #fig, ax = pyplot.subplots(2, 15)
    for i in range(len(all_imgs)):
        all_imgs[i] = (all_imgs[i] * 255).astype(np.uint8)
        img = Image.fromarray(all_imgs[i])
        #ax[i%2][i//2].imshow(img)
        if(i % 10 == 0):
            img.show()

if __name__ == "__main__":
    path_train = 'data/'
    train_volume = create_data(path_train, 'train_v', 30)
    print(train_volume[0][0])
    train_labels = create_data(path_train, 'train_l', 30)
    test_volume = create_data(path_train, 'test_v', 30)
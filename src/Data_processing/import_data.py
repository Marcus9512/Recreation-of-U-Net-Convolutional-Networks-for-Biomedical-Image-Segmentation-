import numpy as np
from PIL import Image
import os


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
                    frame[j,k] = img.getpixel((j, k))

            all_imgs.append(frame)

        except EOFError:
            # Not enough frames in img
            break

    return np.asarray(all_imgs)


if __name__ == "__main__":
    path_train = 'C:\\Users\\Marcu\\Githubs\\dd2424_project\\data\\'
    
    train_volume = create_data(path_train, 'train_v', 30)

    train_labels = create_data(path_train, 'train_l', 30)
    
    test_volume = create_data(path_train, 'test_v', 30)

    np.save(' train_volume.npy', train_volume)
    np.save('train_labels.npy', train_labels)
    np.save('test_volume.npy', test_volume)
    
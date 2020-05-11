import elasticdeform
import imageio
import numpy as np
from PIL import Image


def augment_and_crop(X, Y, reps=1):
    '''
     X.shape = (30, 1, 512, 512)
     Augments image set X with corresponding ground truth labels Y and returns the augmented images
     Creates reps number of new images from each image in X
     '''
    # X = numpy.zeros((200, 300))
    # X[::10, ::10] = 1
    X_deformed = []
    Y_deformed = []
    border_pad = 20
    img_len = 512
    crop = (slice(0+border_pad, img_len-border_pad), slice(0+border_pad, img_len-border_pad))
    for i in range(len(X)):
        # apply deformation with a random 3 x 3 grid and standard dev=10 pixels
        #imageio.imsave('images/in/test_X' + str(i) + ".png", X[i][0])
        #imageio.imsave('images/out/test_Y' + str(i) + ".png", Y[i][0])
        for j in range(reps):
            [x_ij, y_ij] = elasticdeform.deform_random_grid([X[i][0], Y[i][0]], sigma=10, points=3, crop=crop)
            x_ij = x_ij.clip(0,1) #restore default black and white color
            y_ij = y_ij.clip(0,1) #restore default black and white color
            X_deformed += [x_ij]
            Y_deformed += [y_ij]
            #imageio.imsave('images/in/test_X'+str(i)+"_"+str(j) + ".png", x_ij)
            #imageio.imsave('images/out/test_Y'+str(i)+"_"+str(j) + ".png", y_ij)
    X = X[:, :, border_pad:img_len-border_pad, border_pad:img_len-border_pad]
    Y = Y[:, :, border_pad:img_len-border_pad, border_pad:img_len-border_pad]
    X_deformed = np.asarray(X_deformed)
    Y_deformed = np.asarray(Y_deformed)
    X_deformed = np.expand_dims(X_deformed, 1)
    Y_deformed = np.expand_dims(Y_deformed, 1)

    return [X_deformed, Y_deformed]


def augment(reps=5):
    # X = numpy.zeros((200, 300))
    # X[::10, ::10] = 1
    X_deformed = []
    Y_deformed = []
    n = 30

    for i in range(n):
        # apply deformation with a random 3 x 3 grid and standard dev=10 pixels
        #os.mkdir('images')
        #imageio.imsave('images/in/test_X' + str(i) + ".png", X[i][0])
        #imageio.imsave('images/out/test_Y' + str(i) + ".png", Y[i][0])
        x = np.asarray(imageio.imread("data/train/"+str(i)+".jpg"))
        x = x[:, :, 0]
        y = np.asarray(imageio.imread("data/label/"+str(i)+".jpg"))

        print(i)

        for j in range(reps):
            [x_ij, y_ij] = elasticdeform.deform_random_grid([x.astype('float64'), y.astype('float64')], sigma=10, points=3)
            #x_ij = x_ij.clip(0,1) #restore default black and white color
            #y_ij = y_ij.clip(0,1) #restore default black and white color
            x_ij = x_ij.clip(0, 255).astype('uint8')
            y_ij = y_ij.clip(0, 255).astype('uint8')
            X_deformed += [x_ij]
            Y_deformed += [y_ij]
            idx = 31 + i * reps + j
            imageio.imsave('data/train/'+str(idx) + ".jpg", x_ij)
            imageio.imsave('data/label/'+str(idx) + ".jpg", y_ij)

    X_deformed = np.asarray(X_deformed)
    Y_deformed = np.asarray(Y_deformed)
    X_deformed = np.expand_dims(X_deformed, 1)
    Y_deformed = np.expand_dims(Y_deformed, 1)
    return [X_deformed, Y_deformed]
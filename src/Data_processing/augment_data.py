import elasticdeform
import imageio
import numpy as np

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
    border_pad = 20;
    img_len = 512

    crop = (slice(0+border_pad, img_len-border_pad), slice(0+border_pad, img_len-border_pad))
    for i in range(len(X)):
        # apply deformation with a random 3 x 3 grid and standard dev=10 pixels
        imageio.imsave('images/in/test_X' + str(i) + ".png", X[i][0])
        imageio.imsave('images/out/test_Y' + str(i) + ".png", Y[i][0])
        for j in range(reps):
            [x_ij, y_ij] = elasticdeform.deform_random_grid([X[i][0], Y[i][0]], sigma=10, points=3, crop=crop)
            x_ij = x_ij.clip(0,1) #restore default black and white color
            y_ij = y_ij.clip(0,1) #restore default black and white color
            X_deformed += [x_ij]
            Y_deformed += [y_ij]
            imageio.imsave('images/in/test_X'+str(i)+"_"+str(j) + ".png", x_ij)
            imageio.imsave('images/out/test_Y'+str(i)+"_"+str(j) + ".png", y_ij)
    X = X[:, :, border_pad:img_len-border_pad, border_pad:img_len-border_pad]
    Y = Y[:, :, border_pad:img_len-border_pad, border_pad:img_len-border_pad]
    X_deformed = np.asarray(X_deformed)
    Y_deformed = np.asarray(Y_deformed)
    X_deformed = np.expand_dims(X_deformed, 1)
    Y_deformed = np.expand_dims(Y_deformed, 1)
    X = np.append(X, X_deformed, axis=0)
    Y = np.append(Y, Y_deformed, axis=0)
    return X, Y

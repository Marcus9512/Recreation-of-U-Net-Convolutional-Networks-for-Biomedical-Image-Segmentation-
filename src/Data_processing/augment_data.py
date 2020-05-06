import elasticdeform
import imageio
import numpy as np

def augment(X, Y, reps=1):
    '''
     Augments image set X with corresponding ground truth labels Y and returns the augmented images
     '''
    # X = numpy.zeros((200, 300))
    # X[::10, ::10] = 1
    X_deformed = []
    Y_deformed = []

    for i in range(len(X)):
        # apply deformation with a random 3 x 3 grid and standard dev=10 pixels
        for _ in range(reps):
            [xi, yi] = elasticdeform.deform_random_grid([X[i][0], Y[i][0]], sigma=10, points=3)
            X_deformed += [xi]
            Y_deformed += [yi]

    X_deformed = np.asarray(X_deformed)
    Y_deformed = np.asarray(Y_deformed)
    X_deformed = np.expand_dims(X_deformed, 1)
    Y_deformed = np.expand_dims(Y_deformed, 1)
    return [X_deformed, Y_deformed]

    # imageio.imsave('test_X.png', X)
    # imageio.imsave('test_X_deformed.png', X_deformed)
    # imageio.imsave('test_Y_deformed.png', Y_deformed)
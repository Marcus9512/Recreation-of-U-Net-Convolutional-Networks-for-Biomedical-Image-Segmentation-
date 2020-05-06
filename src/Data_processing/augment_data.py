import elasticdeform
import imageio
import numpy as np

def augment(X, Y, reps=1):
    '''
     Augments image set X with corresponding ground truth labels Y and returns the augmented images
     '''
    # X = numpy.zeros((200, 300))
    # X[::10, ::10] = 1

    X_deformed = np.array([])
    Y_deformed = np.array([])

    for i in range(len(X)):
        # apply deformation with a random 3 x 3 grid
        for _ in range(reps):
            [xi, yi] = elasticdeform.deform_random_grid([X[i], Y[i]], sigma=10, points=3)
            np.append(X_deformed, xi)
            np.append(Y_deformed, yi)
    return [X_deformed, Y_deformed]

    # imageio.imsave('test_X.png', X)
    # imageio.imsave('test_X_deformed.png', X_deformed)
    # imageio.imsave('test_Y_deformed.png', Y_deformed)
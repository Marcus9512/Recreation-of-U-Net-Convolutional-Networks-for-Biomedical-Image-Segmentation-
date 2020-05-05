import elasticdeform
import imageio
#import numpy

def augment(X):
    print('test')

    # X = numpy.zeros((200, 300))
    # X[::10, ::10] = 1

    # apply deformation with a random 3 x 3 grid
    X_deformed = elasticdeform.deform_random_grid(X, sigma=10, points=3)

    imageio.imsave('test_X.png', X)
    imageio.imsave('test_X_deformed.png', X_deformed)
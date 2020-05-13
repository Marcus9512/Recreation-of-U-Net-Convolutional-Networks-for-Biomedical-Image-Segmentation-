import elasticdeform
import imageio
import numpy as np
import os, shutil

def augment_and_crop(reps=5):
    '''
        Cropping down each n original images to 4 new (256x256) images, and also augmenting each original
        image to reps new images, resulting in an extended dataset of size n*4 + n*reps.
        The new dataset is saved in data/train and data/label and the original dataset
        is expected to exist in data/train_original and data/label_original
    '''

    if os.path.isdir('data/label'):
        shutil.rmtree('data/label') # FIY these are gitignored
        shutil.rmtree('data/train') # FIY these are gitignored

    os.mkdir('data/label')
    os.mkdir('data/train')

    n = 30
    img_len = 512
    border_pad = 256 // 2
    for i in range(n):
        # apply deformation with a random 3 x 3 grid and standard dev=10 pixels
        x = np.asarray(imageio.imread("data/train_original/"+str(i)+".jpg")).astype('float64')
        y = np.asarray(imageio.imread("data/label_original/"+str(i)+".jpg")).astype('float64')

        print(i)

        for (k, (u, v)) in enumerate(((0,0), (0,256), (256,0), (256, 256))):
            #crop down each original 512x512 image to 4 cropped 256x256 images
            imageio.imsave('data/train/' + str(4*i+k) + ".jpg", x[u:u+256, v:v+256].clip(0, 255).astype('uint8'))
            imageio.imsave('data/label/' + str(4*i+k) + ".jpg", y[u:u+256, v:v+256].clip(0, 255).astype('uint8'))

        for j in range(reps):
            [x0_ij, x1_ij, x2_ij, y_ij] = elasticdeform.deform_random_grid([x[:, :, 0], x[:, :, 1], x[:, :, 2] , y], sigma=10, points=3)
            x_ij = np.zeros((512, 512, 3))
            x_ij[:, :, 0] = x0_ij
            x_ij[:, :, 1] = x1_ij
            x_ij[:, :, 2] = x2_ij
            x_ij = x_ij[border_pad:img_len - border_pad, border_pad:img_len - border_pad] # crop down to 256x256x3
            y_ij = y_ij[border_pad:img_len - border_pad, border_pad:img_len - border_pad] # crop down to 256x256
            x_ij = x_ij.clip(0, 255).astype('uint8')
            y_ij = y_ij.clip(0, 255).astype('uint8')
            idx = 30*4 + i * reps + j
            imageio.imsave('data/train/'+str(idx) + ".jpg", x_ij)
            imageio.imsave('data/label/'+str(idx) + ".jpg", y_ij)
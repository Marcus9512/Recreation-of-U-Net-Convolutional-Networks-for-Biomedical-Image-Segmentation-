from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
import cv2


def mse(A, B):
    err = np.sum((A.astype("float") - B.astype("float")) ** 2)
    err /= float(A.shape[0] * A.shape[1])

    return err


def compare_images(A, B, title):
    m = mse(A, B)
    s = measure.compare_ssim(A, B)

    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
	# show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(A, cmap = plt.cm.gray)
    plt.axis("off")
    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(B, cmap = plt.cm.gray)
    plt.axis("off")
    # show the images
    plt.show()
    
if __name__ == "__main__":
    # load the images -- the original, the original + contrast,
    # and the original + photoshop
    original = cv2.imread("cat.png")
    predict = cv2.imread("Totally_cat.png")
    original = cv2.resize(original, (1000,1000))
    predict = cv2.resize(predict, (1000,1000))
    # convert the images to grayscale
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    predict = cv2.cvtColor(predict, cv2.COLOR_BGR2GRAY)
    # show the figure
    plt.show()
    # compare the images
    compare_images(original, predict, "Original vs. Predicted")


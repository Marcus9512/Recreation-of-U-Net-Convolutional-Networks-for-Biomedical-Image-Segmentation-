import torchvision.datasets as dset
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



# you need to download the coco dataset sepparately
# http://cocodataset.org/#download
# where you can download for example the 2017 version:
# http://cocodataset.org/#detection-2017
# depending on which task you want to perform.


# these paths need to be changed according to your data placements:
def get_coco(path_train_data="", path_train_json="", 
             path_val_data="", path_val_json="", 
             path_test_data="", path_test_json="", test=False):

    coco_train = dset.CocoDetection(root = path_train_data, annFile = path_train_json)
    coco_val = dset.CocoDetection(root = path_val_data, annFile = path_val_json)
    coco_test = dset.CocoDetection(root = path_test_data, annFile = path_test_json)

    if test:
        print("COCO dataset loaded!")
        print("Number of samples in training: ", len(coco_train))
        print("Number of samples in validation: ", len(coco_val))
        print("Number of samples in testing: ", len(coco_test))
    
    return coco_train, coco_val, coco_test


if __name__ == "__main__":

    #Testing if we can load the dataset without errors:
    print("Loading COCO dataset...")
    coco_train, coco_val, coco_test = get_coco(test=True)

    image, target = coco_train[0]

    # testing if we can show the images, so we know they are correct:
    plt.imshow(image)
    plt.show()

    plt.imshow(target)
    plt.show()

import numpy as np

def split_to_training_and_validation(dataset, labels, percent_to_train, percent_to_test):
    assert len(dataset) == len(labels)
    index = np.arange(len(dataset))
    np.random.shuffle(index)

    to_train = int(len(dataset)*percent_to_train)
    to_test = int(len(dataset) * percent_to_test) + to_train
    training = []
    labels_tr = []
    validation = []
    labels_vl = []
    test = []
    labels_test = []

    for i in range(len(dataset)):
        if i <= to_train:
            training.append(dataset[index[i]])
            labels_tr.append(labels[index[i]])
        elif i <= to_test:
            test.append(dataset[index[i]])
            labels_test.append(labels[index[i]])
        else:
            validation.append(dataset[index[i]])
            labels_vl.append(labels[index[i]])

    return training, labels_tr, validation, labels_vl, test, labels_test

def rand_error(prediction, target):
    '''
    Using the formula from here: https://imagej.net/Rand_error
    '''
    iflat = prediction.view(-1)
    tflat = target.view(-1)
    true_positive = 0
    true_negative = 0
    n = len(iflat)
    for i in range(n):
        if iflat[i] == 1 and tflat[i] == 1:
            true_positive += 1
        elif iflat[i] == 0 and tflat[i] == 0:
            true_negative += 1
    return 1 - (true_positive+true_negative) / (n*(n-1)/2)


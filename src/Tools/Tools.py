import numpy as np

def split_to_training_and_validation(dataset, labels, percent_to_train):
    assert len(dataset) == len(labels)
    index = np.arange(len(dataset))
    np.random.shuffle(index)

    to_train = int(len(dataset)*percent_to_train)
    training = []
    labels_tr = []
    validation = []
    labels_vl = []

    for i in range(len(dataset)):
        if i <= to_train:
            training.append(dataset[index[i]])
            labels_tr.append(labels[index[i]])
        else:
            validation.append(dataset[index[i]])
            labels_vl.append(labels[index[i]])
    return training, labels_tr, validation, labels_vl
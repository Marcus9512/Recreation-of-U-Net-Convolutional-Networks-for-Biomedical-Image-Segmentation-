'''
Test the dummy file
'''

from src.Data_processing.import_data import *

def test_import_data():
    '''
    Checks the shape of the output from create_data
    :return:
    '''
    path_train = "data/"

    res = create_data(path_train, 'train_v', 30)
    assert res.shape == (30, 1, 512, 512)
    res = create_data(path_train, 'train_l', 30)
    assert res.shape == (30, 1, 512, 512)
    res = create_data(path_train, 'test_v', 30)
    assert res.shape == (30, 1, 512, 512)

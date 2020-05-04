'''
Test the dummy file
'''

from src.check_travis.dummy import *
from import_data import create_data

def test_dummy_function():


    path_train = 'C:\\Users\\Marcu\\Githubs\\dd2424_project\\data\\'
    
    res = create_data(path_train, 'train_v', 30)
    assert res.shape == (30, 512, 512)
    res = create_data(path_train, 'train_l', 30)
    assert res.shape == (30, 512, 512)
    res = create_data(path_train, 'test_v', 30)
    assert res.shape == (30, 512, 512)
    res = create_data()

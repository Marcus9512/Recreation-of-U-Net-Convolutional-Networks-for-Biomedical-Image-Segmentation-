'''
Test the dummy file
'''

from src.check_travis.dummy import *

def test_dummy_function():

    res = return_one()
    assert res == 1

    res = return_two()
    assert res == 2

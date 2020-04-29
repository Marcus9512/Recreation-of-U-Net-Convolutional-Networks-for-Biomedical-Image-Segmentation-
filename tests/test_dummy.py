'''
Test the dummy file
'''

from src.Dummy import Dummy

def test_dummy_function():

    res = Dummy.return_one()
    assert res == 1

    res = Dummy.return_two()

    assert res == 2
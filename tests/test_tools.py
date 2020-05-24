from src.Tools.Tools import *

def test_rand_error():
    pred = np.ones((256,256))
    target = np.ones((256, 256))
    out = rand_error_2(pred, target)
    print(out)
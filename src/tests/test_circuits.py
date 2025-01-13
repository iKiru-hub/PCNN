import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import libs.pclib
import pytest
import numpy as np


def test_add():
    a = 1+1
    assert a == 2


if __name__ == '__main__':

    add()
    print("All tests passed")



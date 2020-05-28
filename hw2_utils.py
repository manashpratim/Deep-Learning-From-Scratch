import importlib
import unittest
from collections import namedtuple
from datetime import datetime
import inspect
import gzip
import pickle as pkl
import os
import numpy as np
import tensorflow as tf

import warnings
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

Splits = namedtuple("Splits", ["train", "valid", "test"])
Domains = namedtuple("Domains", ["X", "Y"])


class TestCase(unittest.TestCase):
    def setUp(self):
        if self.module_name:
            self.solution = importlib.import_module(self.module_name)

        unittest.TestCase.setUp(self)

    @staticmethod
    def assertArrayEqual(solution, submission, msg='Arrays not equal.'):
        return np.testing.assert_array_almost_equal(
            solution, submission,
            decimal=6,
            err_msg=msg
        )

    def assertShapeEqual(self, solution, submission, msg='Shapes not equal.'):
        return self.assertTupleEqual(
            solution, submission,
            msg=msg
        )


def run_tests(testclass):
    tests = unittest.TestLoader().loadTestsFromTestCase(testclass)

    f = open(os.devnull, "w")

    runner = unittest.TextTestRunner(stream=f,
                                     descriptions=False,
                                     verbosity=0,
                                     buffer=False)

    for test in tests:
        print(f"=== BEGIN {test} ===")
        res = runner.run(test)

        bads = [m[1] for m in [*res.errors, *res.failures]]

        if len(bads) == 0:
            print("OK")
        else:
            print("FAIL")

        for bad in bads:
            print(bad)

        print(f"=== END {test} ===")

    f.close()


def accuracy(model, x, y):
    return np.mean(np.argmax(model.predict(x), axis=1) == y)


def load_mnist():
    for d in ['.', '..', '../..', '/autograder/source/tests']:
        try:
            path = os.path.join(d, "mnist.pkl.gz")
            assert (os.path.exists(path) and os.access(path, os.R_OK))
            break
        except Exception:
            continue

    with gzip.open(path, 'rb') as f:
        (train_X, train_y), \
        (val_X, val_y), \
        (test_X, test_y) = pkl.load(f, encoding='latin1')

    return Splits(Domains(train_X, train_y), Domains(val_X, val_y),
                  Domains(test_X, test_y))


def timeit(thunk):
    start = datetime.now()

    ret = thunk()

    return ret, datetime.now() - start


def exercise(andrew_username=None, seed=None):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    cname = inspect.getouterframes(inspect.currentframe(), 2)[1][3]

    seedh = hash(hash(andrew_username) + hash(cname) + hash(seed)) % 0xffffffff

    np.random.seed(seedh)
    tf.compat.v1.set_random_seed(seedh)

    print(f"=== EXERCISE {cname} {andrew_username} {seed:x} {seedh:x} ===")

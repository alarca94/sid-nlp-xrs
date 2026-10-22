import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import unittest

from nlp_xrs.run import run_experiment
from tests.test_model.mini_base import MiniSandbox

SANDBOX = MiniSandbox()


def setUpModule():
    SANDBOX.setup()


def tearDownModule():
    SANDBOX.teardown()


class TestGenerativeRecommender(unittest.TestCase):
    def test_nrt(self):
        run_experiment(SANDBOX.args("nrt"))

    def test_peter(self):
        run_experiment(SANDBOX.args("peter"))

    def test_sequer(self):
        run_experiment(SANDBOX.args("sequer"))

    def test_sequertextfusion(self):
        run_experiment(SANDBOX.args("sequertextfusion"))


if __name__ == "__main__":
    unittest.main()

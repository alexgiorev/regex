import unittest
import random

from patterns import *


class TestChar(unittest.TestCase):
    def setUp(self):
        char = chr(random.randint(0, 127))
        p = Char

    def matchtest(self, m):
        pass

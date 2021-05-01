import unittest

import HashDL


class TestSGD(unittest.TestCase):
    def test_SGD(self):
        sgd = HashDL.SGD()


class TestAdam(unittest.TestCase):
    def test_Adam(self):
        adam = HashDL.Adam()


class TestWTA(unittest.TestCase):
    def test_WTA(self):
        wta = HashDL.WTA(8, 32)



class TestDWTA(unittest.TestCase):
    def test_DWTA(self):
        dwta = HashDL.DWTA(8, 32)


class TestConstantFrequency(unittest.TestCase):
    def test_ConstantFrequency(self):
        cf = HashDL.ConstantFrequency(50)


class TestExponentialDecay(unittest.TestCase):
    def test_ExponentialDecay(self):
        ed = HashDL.ExponentialDecay(50, 1e-5)

if __name__ == "__main__":
    unittest.main()

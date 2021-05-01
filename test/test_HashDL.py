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

    def test_invalid_type_str(self):
        with self.assertRaises(TypeError):
            cf = HashDL.ConstantFrequency("str")

    def test_invalid_type_None(self):
        with self.assertRaises(TypeError):
            cf = HashDL.ConstantFrequency(None)


class TestExponentialDecay(unittest.TestCase):
    def test_ExponentialDecay(self):
        ed = HashDL.ExponentialDecay(50, 1e-5)

    def test_invalid_type_str(self):
        with self.assertRaises(TypeError):
            ed = HashDL.ExponentialDecay("abc", 1e-5)

        with self.assertRaises(TypeError):
            ed = HashDL.ExponentialDecay(50, "abs")


    def test_invalud_type_None(self):
        with self.assertRaises(TypeError):
            ed = HashDL.ExponentialDecay(None, 1e-5)

        with self.assertRaises(TypeError):
            ed = HashDL.ExponentialDecay(50, None)

if __name__ == "__main__":
    unittest.main()

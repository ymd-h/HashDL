import unittest

import HashDL


class TestSGD(unittest.TestCase):
    def test_SGD(self):
        sgd = HashDL.SGD()

    def test_invalid_type_str(self):
        with self.assertRaises(TypeError):
            sgd = HashDL.SGD("str")

    def test_invalid_type_None(self):
        with self.assertRaises(TypeError):
            sgd = HashDL.SGD(None)

    def test_negative_learning_rate(self):
        with self.assertRaises(ValueError):
            sgd = HashDL.SGD(-10)


class TestAdam(unittest.TestCase):
    def test_Adam(self):
        adam = HashDL.Adam()

    def test_invalid_type_str(self):
        with self.assertRaises(TypeError):
            adam = HashDL.Adam("str")

    def test_invalid_type_None(self):
        with self.assertRaises(TypeError):
            adam = HashDL.Adam(None)

    def test_negative_learning_rate(self):
        with self.assertRaises(ValueError):
            adam = HashDL.Adam(-10)


class TestWTA(unittest.TestCase):
    def test_WTA(self):
        wta = HashDL.WTA(32, 8)

    def test_invalid_type_str(self):
        with self.assertRaises(TypeError):
            wta = HashDL.WTA("act", 8)

        with self.assertRaises(TypeError):
            wta = HashDL.WTA(8, "32")

    def test_invalid_type_None(self):
        with self.assertRaises(TypeError):
            wta = HashDL.WTA(None, 8)

        with self.assertRaises(TypeError):
            wta = HashDL.WTA(8, None)


class TestDWTA(unittest.TestCase):
    def test_DWTA(self):
        dwta = HashDL.DWTA(8, 8)

    def test_invalid_type_str(self):
        with self.assertRaises(TypeError):
            dwta = HashDL.DWTA("act", 8)

        with self.assertRaises(TypeError):
            dwta = HashDL.DWTA(8, "32")

    def test_invalid_type_None(self):
        with self.assertRaises(TypeError):
            dwta = HashDL.DWTA(None, 8)

        with self.assertRaises(TypeError):
            dwta = HashDL.DWTA(8, None)


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


class TestNetwork(unittest.TestCase):
    def test_default_network(self):
        net = HashDL.Network(16)

    def test_negetive_intput_size(self):
        with self.assertRaises(ValueError):
            net = HashDL.Network(-2)

    def test_invalid_type_str_input_size(self):
        with self.assertRaises(TypeError):
            net = HashDL.Network("abc")

    def test_invalid_type_None(self):
        with self.assertRaises(TypeError):
            net = HashDL.Network(None)

    def test_negative_units(self):
        with self.assertRaises(ValueError):
            net = HashDL.Network(16, [-10, 2, 5])

    def test_negative_L(self):
        with self.assertRaises(ValueError):
            net = HashDL.Network(16, (10, 10), -3)

    def test_SGD(self):
        net = HashDL.Network(16, optimizer=HashDL.SGD())

    def test_Adam(self):
        net = HashDL.Network(16, optimizer=HashDL.Adam())

    def test_WTA(self):
        net = HashDL.Network(16, hash=HashDL.WTA(8, 8))

    def test_DWTA(self):
        net = HashDL.Network(16, hash=HashDL.DWTA(8, 8))

    def test_ConstantFrequency(self):
        net = HashDL.Network(16, scheduler=HashDL.ConstantFrequency(50))

    def test_ExponentialDecay(self):
        net = HashDL.Network(16, scheduler=HashDL.ExponentialDecay(50, 1e-3))


if __name__ == "__main__":
    unittest.main()

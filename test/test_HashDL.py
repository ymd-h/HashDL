import unittest

import numpy as np

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

class TestActivation(unittest.TestCase):
    def test_Linear(self):
        act = HashDL.Linear()
        self.assertEqual(act(0.0), 0.0)
        self.assertEqual(act(1.0), 1.0)
        self.assertEqual(act(-1.0), -1.0)

    def test_ReLU(self):
        act = HashDL.ReLU()
        self.assertEqual(act(0.0), 0.0)
        self.assertEqual(act(1.0), 1.0)
        self.assertEqual(act(-1.0), 0.0)

    def test_Sigmoid(self):
        act = HashDL.Sigmoid()
        self.assertEqual(act(0.0), 0.5)
        self.assertLessEqual(act(100), 1.0)
        self.assertGreaterEqual(act(-100), 0.0)

class TestInitializer(unittest.TestCase):
    def test_ConstantInitializer(self):
        init = HashDL.ConstantInitializer(5)
        self.assertEqual(init(), 5)

    def test_GaussianInitializer(self):
        init = HashDL.GaussInitializer(0, 1.0)
        init()

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

    def test_call(self):
        data_size = 2
        batch_size = 1

        net = HashDL.Network(data_size, units=(1,), L = 5,
                             optimizer = HashDL.Adam(),
                             scheduler = HashDL.ConstantFrequency(1),
                             hash = HashDL.DWTA(8, 1))

        X = np.zeros((batch_size, data_size))
        Y = net(X)
        self.assertIsInstance(Y, np.ndarray)
        self.assertEqual(Y.shape, (batch_size, 1))

    def test_backward(self):
        data_size = 2
        batch_size = 1

        net = HashDL.Network(data_size, units=(1,), L = 5,
                             optimizer = HashDL.Adam(),
                             scheduler = HashDL.ConstantFrequency(1),
                             hash = HashDL.DWTA(8, 1))

        X = np.zeros((batch_size, data_size))
        Y = net(X)
        net.backward(Y)

if __name__ == "__main__":
    unittest.main()

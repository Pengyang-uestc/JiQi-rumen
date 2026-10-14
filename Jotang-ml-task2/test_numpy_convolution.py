import unittest
import numpy as np
from numpy_convolution import conv2d


class ConvolutionTests(unittest.TestCase):
    def test_direction(self):
        image = np.arange(1, 10).reshape(3, 3)
        kernel = [[1, 2], [3, 4]]
        np.testing.assert_array_equal(conv2d(image, kernel, "valid"), [[23, 33], [53, 63]])
        np.testing.assert_array_equal(conv2d(image, kernel, "valid", flip_kernel=False),
                                      [[37, 47], [67, 77]])

    def test_padding_and_stride(self):
        image = np.ones((3, 3))
        expected = [[4, 6, 4], [6, 9, 6], [4, 6, 4]]
        np.testing.assert_array_equal(conv2d(image, np.ones((3, 3))), expected)
        np.testing.assert_array_equal(conv2d(image, np.ones((3, 3)), stride=2), [[4, 4], [4, 4]])
        np.testing.assert_array_equal(conv2d(image, np.ones((2, 2))), [[4, 4, 2], [4, 4, 2], [2, 2, 1]])

    def test_rgb_and_float(self):
        rgb = np.full((2, 2, 3), [10, 100, 255], dtype=np.uint8)
        np.testing.assert_array_equal(conv2d(rgb, [[1]]), rgb)
        np.testing.assert_array_equal(conv2d(rgb, [[-2]]), rgb.astype(float) * -2)
        self.assertEqual(conv2d(rgb, [[2]]).dtype, np.float64)
        self.assertEqual(conv2d(rgb, [[2]])[0, 0, 2], 510)

    def test_invalid(self):
        for kwargs in [dict(stride=0), dict(stride=True), dict(padding="wrong")]:
            with self.assertRaises(ValueError):
                conv2d(np.ones((2, 2)), [[1]], **kwargs)
        with self.assertRaises(ValueError):
            conv2d(np.ones((2, 2, 4)), [[1]])
        with self.assertRaises(ValueError):
            conv2d(np.ones((2, 2)), np.ones((3, 3)), "valid")


if __name__ == "__main__":
    unittest.main()

import unittest
import os
import sys

# Add the project root to the Python path (no print statements)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import the helper functions
from src.utils.helpers import calculate_moving_average, normalize_data, split_data

class TestUtils(unittest.TestCase):

    def test_calculate_moving_average(self):
        data = [1, 2, 3, 4, 5]
        # Test with valid window size
        result = calculate_moving_average(data, 3)
        expected = (3 + 4 + 5) / 3
        self.assertEqual(result, expected)
        
        # Test with window size larger than data
        result = calculate_moving_average(data, 6)
        self.assertIsNone(result)

    def test_normalize_data(self):
        data = [1, 2, 5, 10]
        result = normalize_data(data)
        expected = [0.0, 1/9, 4/9, 1.0]
        
        # Test each value with a small tolerance for floating point errors
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e, places=10)
    
    def test_split_data(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        train, test = split_data(data, 0.7)
        
        self.assertEqual(len(train), 7)
        self.assertEqual(len(test), 3)
        self.assertEqual(train, [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(test, [8, 9, 10])

if __name__ == '__main__':
    unittest.main()
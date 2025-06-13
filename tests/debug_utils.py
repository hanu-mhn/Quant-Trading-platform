"""
Debug script to check if the utils tests are working
"""
import sys
import os

# Print Python path
print(f"Python path: {sys.path}")

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
print(f"Added to path: {project_root}")
print(f"Updated Python path: {sys.path}")

# Try importing the modules
try:
    from src.utils.helpers import calculate_moving_average, normalize_data, split_data
    print("Successfully imported helper functions!")
    
    # Test calculate_moving_average
    data = [1, 2, 3, 4, 5]
    result = calculate_moving_average(data, 3)
    expected = (3 + 4 + 5) / 3
    print(f"calculate_moving_average([1,2,3,4,5], 3) = {result}, expected {expected}")
    print(f"Test passed: {result == expected}")
    
    # Test normalize_data
    data = [1, 2, 5, 10]
    result = normalize_data(data)
    expected = [0.0, 1/9, 4/9, 1.0]
    print(f"normalize_data([1,2,5,10]) = {result}")
    print(f"expected = {expected}")
    
    # Test split_data
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    train, test = split_data(data, 0.7)
    print(f"split_data([1..10], 0.7) = {train}, {test}")
    print(f"Expected: {[1,2,3,4,5,6,7]}, {[8,9,10]}")
    
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Other error: {e}")

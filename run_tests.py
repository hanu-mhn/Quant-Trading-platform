"""
Run pytest tests and save results to a file
"""
import os
import sys
import subprocess

# Print current working directory
current_dir = os.getcwd()
print(f"Current directory: {current_dir}")

# Path to save results
results_file = os.path.join(current_dir, "test_results.txt")

# Run pytest and save output
try:
    # Ensure we're in the right directory
    os.chdir("d:\\QUANT\\QT_python\\quant-trading-platform")
    print(f"Changed to directory: {os.getcwd()}")
    
    # Run pytest with verbose flag
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
    print(f"Running command: {' '.join(cmd)}")
    
    # Execute pytest and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Write output to file
    with open(results_file, 'w') as f:
        f.write(f"STDOUT:\n{result.stdout}\n\n")
        f.write(f"STDERR:\n{result.stderr}\n\n")
        f.write(f"Return code: {result.returncode}")
    
    print(f"Test results written to {results_file}")
    print(f"Return code: {result.returncode}")
    
except Exception as e:
    with open(results_file, 'w') as f:
        f.write(f"ERROR:\n{str(e)}")
    print(f"Error: {str(e)}")

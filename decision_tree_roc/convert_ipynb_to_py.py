import os
import sys
import subprocess

def convert_ipynb_to_py(ipynb_path):
    if not os.path.exists(ipynb_path):
        print(f"File not found: {ipynb_path}")
        return
    
    if not ipynb_path.endswith('.ipynb'):
        print("Please provide a .ipynb file.")
        return

    try:
        subprocess.run(['jupyter', 'nbconvert', '--to', 'script', ipynb_path], check=True)
        print("Conversion completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_ipynb_to_py.py <notebook_file.ipynb>")
    else:
        convert_ipynb_to_py(sys.argv[1])

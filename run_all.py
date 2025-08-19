import os
import subprocess
import sys

# Path where to start searching (project root)
ROOT_DIR = "."

# Collect all .py files recursively
py_files = []
for root, dirs, files in os.walk(ROOT_DIR):
    for f in files:
        if f.endswith(".py") and f not in ["run_all.py", "sentiment_analysis_fix.py"]:
            py_files.append(os.path.join(root, f))

print("Discovered Python files:")
for f in py_files:
    print(" -", f)

# Run them one by one
for file in py_files:
    print(f"\n>>> Running {file}...\n")
    subprocess.run([sys.executable, file])

#!/usr/bin/env python3
"""
Execute the GALI Analysis Jupyter Notebook
"""

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys

def execute_notebook(input_path, output_path):
    """Execute a Jupyter notebook and save the result."""
    print(f"Loading notebook: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    print("Executing notebook...")
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    try:
        ep.preprocess(nb, {'metadata': {'path': '.'}})
        print("✓ Notebook executed successfully")

        # Save executed notebook
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"✓ Saved executed notebook to: {output_path}")

        return True
    except Exception as e:
        print(f"✗ Error executing notebook: {e}")
        # Save partial results anyway
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"✓ Saved partial results to: {output_path}")
        return False

if __name__ == "__main__":
    input_nb = "GALI_Analysis_Report.ipynb"
    output_nb = "GALI_Analysis_Report_Executed.ipynb"

    success = execute_notebook(input_nb, output_nb)
    sys.exit(0 if success else 1)

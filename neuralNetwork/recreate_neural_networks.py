import pathlib
import os
from nbconvert import PythonExporter
import nbformat
 
path_to_file = pathlib.Path(__file__)
path_to_folder = path_to_file.parents[0]

for file in path_to_folder.iterdir():
    if file.name.startswith("neural_network") and file.name.endswith(".ipynb"):
        with open(file) as f:
            nb = nbformat.read(f, as_version=4)

        # Initialize the exporter
        exporter = PythonExporter()

        # Extract the Python code from the notebook
        body, resources = exporter.from_notebook_node(nb)

        # Execute the Python code
        exec(body)

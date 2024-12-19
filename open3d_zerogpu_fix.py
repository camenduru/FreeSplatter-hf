import fileinput
import site
from pathlib import Path

with fileinput.FileInput(f'{site.getsitepackages()[0]}/open3d/__init__.py', inplace=True) as file:
    for line in file:
        print(line.replace('_pybind_cuda.open3d_core_cuda_device_count()', '1'), end='')

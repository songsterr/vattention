import os
from setuptools import Extension, setup
import torch
from torch.utils import cpp_extension

# Use dynamic PyTorch library detection for better compatibility
include_dirs = cpp_extension.include_paths() + ['.']

# Get library paths from torch
library_dirs = cpp_extension.library_paths()

setup(name='vattention',
      version='0.0.1',
      ext_modules=[cpp_extension.CUDAExtension('vattention', ['vattention.cu'],
      include_dirs=include_dirs,
      library_dirs=library_dirs,
      extra_link_args=['-lc10', '-lcuda', '-ltorch'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension}
      )


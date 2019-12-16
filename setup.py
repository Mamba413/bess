import re
import os
import sys
import numpy
from setuptools import setup, find_packages, Extension
from os import path

os_type = 'MS_WIN64'
if sys.platform.startswith('win32'):
    python_path = sys.base_prefix
    temp = python_path.split("\\")
    version = str(sys.version_info.major) + str(sys.version_info.minor)
    CURRENT_DIR = os.path.dirname(__file__)
    path1 = "-I" + python_path + "\\include"
    path2 = "-L" + python_path + "\\libs"
    os.system('bash pre.sh ' + python_path + ' ' + version)

    cbess_module = Extension(name='BeSS._cbess',
                          sources=['src/bess_lm.cpp', 'src/List.cpp', 'src/utilities.cpp', 'src/normalize.cpp', 'src/bess.i',
                                   'src/bess_cox.cpp', 'src/bess_cox_group.cpp', 'src/bess_glm.cpp', 'src/bess_glm_group.cpp',
                                   'src/logistic.cpp', 'src/tmp.cpp', 'src/coxph.cpp'],
                          language='c++',
                          extra_compile_args=["-DNDEBUG", "-fopenmp", "-O2", "-Wall", "-std=c++11", "-mtune=generic", "-D%s" % os_type, path1, path2],
                          extra_link_args=['-lgomp'],
                          libraries=["vcruntime140"],
                          include_dirs = ["BeSS", numpy.get_include()],
                          swig_opts=["-c++"]
                          )

else:
    cbess_module = Extension(name='BeSS._cbess',
                          sources=['src/bess_lm.cpp', 'src/List.cpp', 'src/utilities.cpp', 'src/normalize.cpp', 'src/bess.i'],
                          language='c++',
                          extra_compile_args=["-DNDEBUG", "-fopenmp", "-O2", "-Wall", "-std=c++11"],
                          extra_link_args=['-lgomp'],
                          libraries=["vcruntime140"],
                          include_dirs = ["BeSS", numpy.get_include()],
                          swig_opts=["-c++"]
                          )

setup(name='BeSS',
      version='1.0.0',
      description='BeSS',
      packages=find_packages(),
      ext_modules=[cbess_module]
      )



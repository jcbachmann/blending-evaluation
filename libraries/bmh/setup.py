import os
import platform
import re
import subprocess
import sys
from distutils.version import LooseVersion
from typing import Optional

from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext

with open('README.md', 'r') as fh:
    long_description = fh.read()


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = '', target: Optional[str] = None):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.target = target


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                'CMake must be installed to build the following extensions: ' +
                ', '.join(e.name for e in self.extensions)
            )

        if platform.system() == 'Windows':
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError('CMake >= 3.1.0 is required on Windows')

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext: CMakeExtension):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir, '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        if ext.target:
            build_args += ['--target', ext.target]

        if platform.system() == 'Windows':
            cmake_args += [f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}']
            if sys.maxsize > 2 ** 32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = f'{env.get("CXXFLAGS", "")} -DVERSION_INFO=\\"{self.distribution.get_version()}\\"'
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


setup(
    name='bmh',
    version='0.0.1',
    author='Michael Cipold',
    author_email='michael@cipold.de',
    description='Bulk Material Homogenization',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jcbachmann/blending-evaluation',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'bokeh',
        'dask',
        'distributed',
        'dash',
        'dash-html-components',
        'dash-core-components',
        'dask',
        'distributed',
        # 'jmetalpy',  # TODO
        'matplotlib',
        'numpy',
        'pandas',
        'plotly',
        'tornado',
    ],
    ext_modules=[
        CMakeExtension(
            'blending_simulator_lib',
            sourcedir='../../../BlendingSimulator',  # https://github.com/jcbachmann/blending-simulation
            target='blending_simulator_lib'
        )
    ],
    cmdclass=dict(build_ext=CMakeBuild),
)

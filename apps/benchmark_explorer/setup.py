import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='benchmark_explorer',
    version='0.0.1',
    author='Michael Cipold',
    author_email='michael@cipold.de',
    description='Benchmark Explorer',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jcbachmann/blending-evaluation',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'bmh',
        'data_explorer',
        'scipy',
    ],
)

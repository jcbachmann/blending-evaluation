import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='plant_simulator',
    version='0.0.1',
    author='Michael Cipold',
    author_email='michael@cipold.de',
    description='Plant Simulator',
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
        'bokeh',
        'matplotlib',
        'numpy',
        'pandas',
        'tornado',
    ],
)

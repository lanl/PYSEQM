from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='PYSEQM',
    author='Guoqing Zhou',
    author_email='guoqingz@usc.edu',
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.0.1',

    description='A Semi-Empirical Born Oppenheimer Molecular Dynamics Engine implemented in PyTorch.',
    long_description=long_description,
    
    # The project's main homepage.
    url='https://github.com/lanl/PYSEQM',

    # Choose your license
    license='BSD 3-Clause License',

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Research',
        'Topic :: Quantum Chemistry',

        # Pick your license as you wish (should match "license" above)
        'License :: BSD 3-Clause License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3'
        'Programming Language :: Python :: 3.7'
        'Programming Language :: Python :: 3.8'
    ],

    # What does your project relate to?
    keywords='Semi-Empirical Quantum Mechanics',

    packages=find_packages(exclude=['params', 'tests']),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['numpy', 'torch>=1.2'],
    include_package_data=True,
    package_data={'seqm': ['params/*.csv']},

)
from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Module to generate jacobian matrix.'
LONG_DESCRIPTION = 'Python function that creates a jacobian matrix from list of functions and boundry conditions.'

# Setting up
setup(
        name="src", 
        version=VERSION,
        author="Marcel Čarman, Nejc Jeraj",
        author_email="<marcelcarman1@gmail.com>,<nejcjeraj@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[numpy, sympy],
        
        keywords=['python', 'jacobian matrix', 'numerical jacobian', 'analytical jacobian'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)

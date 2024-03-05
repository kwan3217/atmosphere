"""
Setup file for the atmosphere package.

The only required fields for setup are name, version, and packages. Other fields to consider (from looking at other
projects): keywords, include_package_data, requires, tests_require, package_data
"""
from setuptools import setup

setup(
    name='atmosphere',
    version='0.1.0',
    description='Atmosphere model appropriate for aerospace vehicle simulations',
    url='https://github.com/kwan3217/atmosphere/',
    author='kwan3217',
    author_email='kwan3217@gmail.com',
    license='BSD 2-clause',
    packages=['atmosphere'],
    python_requires='>=3.10, <4',
    install_requires=[
                      'numpy',
                     ],

)

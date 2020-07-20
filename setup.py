"""
The build/compilations setup

>> pip install -r requirements.txt
>> python setup.py install
"""
from setuptools import setup
setup(
    name='instanse_seg',
    version='1.0',
    description='instance segmentation tutorial',
    author='Hudanyun Sheng',
    packages=['instanse_seg'],
    install_requires=['tensorflow', 'keras']
)

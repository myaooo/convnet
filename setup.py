"""
Setup scipt
"""
from setuptools import setup, find_packages

setup(
    name="cnn",
    version="0.0.0",
    author="MING Yao",
    author_email="yaoming.thu@gmail.com",
    description="CNN implementation based on Tensorflow",
    keywords="cnn, vislab, hkust",
    url="https://github.com/myaooo/convnet",
    packages=find_packages(),
    install_requires=[
    ],
    entry_points={
        'console_scripts': ['convnet = py.main:main']
    }
)

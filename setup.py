import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="5g_deep_learning",

    description="Modules to run an intrusion detection system on KDD daatset using stacked denoising auto-encoders.",

    author="Filippo Pedrazzini",

    packages=find_packages(exclude=['model', 'results', 'tensorboard', 'notebooks', 'data', 'tableau']),

    long_description=read('README.md'),
)
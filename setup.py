from setuptools import setup, find_packages

setup(
    name='omni-dreamer',
    version='0.0.1',
    description='Diverse Plausible 360-Degree Image Outpainting for Efficient 3DCG Background Creation',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)

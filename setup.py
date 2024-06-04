from setuptools import setup

setup(
    name="vocalocator",
    version="1.0.0",
    description="Tool for sound-source localization of rodent vocalizations",
    url="https://github.com/neurostatslab/vocalocator",
    author="NeuroStats Lab",
    license="MIT",
    packages=["vocalocator", "vocalocator.training", "vocalocator.architectures"],
    install_requires=["h5py", "numpy", "torch"],
)

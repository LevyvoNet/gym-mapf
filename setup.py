import setuptools
import os

with open(os.path.join(os.path.dirname(__file__), "README.md")) as readme:
    long_description = readme.read()

setuptools.setup(
    name="gym-mapf",
    version="0.0.13",
    author="LevyvoNet",
    author_email="eladlevy2@gmail.com",
    description="Multi-Agent Path Finding gym environment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LevyvoNet/gym-mapf",
    packages=setuptools.find_packages(),
    include_package_data=True,
    test_suite="gym_mapf.tests",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

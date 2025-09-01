from setuptools import setup, find_packages

setup(
    name="construction-material-classifier",
    version="1.0.0",
    author="Rahul Reddy",
    author_email="rahul.reddy@example.edu",
    description="Lightweight deep learning for construction material classification on edge devices",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rahulreddy/construction-material-classifier",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.2",
        "torchvision>=0.16.2",
        "numpy>=1.26.3",
        "pandas>=2.1.4",
        "scikit-learn>=1.4.0",
        "matplotlib>=3.8.2",
        "Pillow>=10.2.0",
        "tqdm>=4.66.1",
        "PyYAML>=6.0.1",
    ],
)

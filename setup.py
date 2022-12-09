from setuptools import find_packages, setup

setup(
    name="cqt-pytorch",
    packages=find_packages(exclude=[]),
    version="0.0.5",
    license="MIT",
    description="CQT Pytorch",
    long_description_content_type="text/markdown",
    author="Flavio Schneider",
    author_email="archinetai@protonmail.com",
    url="https://github.com/archinetai/cqt-pytorch",
    keywords=["artificial intelligence", "deep learning"],
    install_requires=[
        "torch>=1.6",
        "data-science-types>=0.2",
        "einops>=0.4",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)

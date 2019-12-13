import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name="ProtConv2D",
    version="0.1",
    scripts=[
        "ProtConv2D/bin/kcc_results_for_pub.py",
        "ProtConv2D/bin/pdb2image.py",
        "ProtConv2D/bin/train_cath_classifier.py",
        "ProtConv2D/bin/cath_images_to_hdf5.py",
        "ProtConv2D/bin/plot_model.py",
    ],
    author="Tobias Sikosek",
    author_email="toby.sikosek@gmail.com",
    description="Create protein structure embeddings with 2D ConvNets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tobysiko/protconv",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

## Star-galaxy Classification Using Deep Convolutional Neural Networks

- Edward J Kim
- Robert J Brunner

### Paper

The LaTex source files are in the `paper` directory.

### IPython/Jupyter Notebooks

[How to generate SDSS cutout images](notebooks/fetch_sdss.ipynb):
This notebook demonstrates how to generate cutout images from the SDSS database.
Note: For demonstration and testing purposes only.
For creating a training set, use
[scripts/fetch_sdss.py](scripts/fetch_sdss.py).

[Training ConvNets](notebooks/convnet.ipynb):
This notebook demonstrates the ConvNet architecture with a sample training set
of 100 images.
For demonstration and testing purposes only.
In the paper, we used a much larger training set.
See [scripts/train_cnn.py](scripts/train_cnn.py).

### Dockerfile

The Dockerfile for the Jupyter notebook server that has been used to run
the notebooks is provided in the `docker` directory.

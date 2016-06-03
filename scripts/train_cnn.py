'''
If you are using a GPU, write the following in ~/.theanorc.

[global]
device=gpu
floatX=float32

[blas]
ldflags=-lopenblas

[cuda]
root=/opt/apps/cuda/7.0

[nvcc]
fastmath=True
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import time
import cPickle as pickle

import numpy as np

import theano
import theano.tensor as T

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.objectives import categorical_crossentropy
from lasagne.nonlinearities import leaky_rectify
from lasagne.init import Orthogonal, Constant
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
from lasagne.nonlinearities import softmax

import bmc

X = np.load("../data/sdss_training_images.npy")
print("X.shape = {}, X.min = {}, X.max = {}".format(X.shape, X.min(), X.max()))

y = np.load("../data/sdss_training_labels.npy")
print("y.shape = {}, y.min = {}, y.max = {}".format(y.shape, y.min(), y.max()))

def renormalize(array):
    return (array - array.min()) / (array.max() - array.min())

for i in range(5):
    X[:, i, :, :] = renormalize(X[:, i, :, :])
    
y = renormalize(y).astype(np.int32)
print("X.shape = {}, X.min = {}, X.max = {}".format(X.shape, X.min(), X.max()))
print("y.shape = {}, y.min = {}, y.max = {}".format(y.shape, y.min(), y.max()))

def compute_PCA(array):

    nimages0, nchannels0, height0, width0 = array.shape
    rolled = np.transpose(array, (0, 2, 3, 1))
    # transpose from N x channels x height x width  to  N x height x width x channels
    nimages1, height1, width1, nchannels1 = rolled.shape
    # check shapes
    assert nimages0 == nimages1
    assert nchannels0 == nchannels1
    assert height0 == height1
    assert width0 == width1
    # flatten
    reshaped = rolled.reshape(nimages1 * height1 * width1, nchannels1)
    
    from sklearn.decomposition import PCA
    
    pca = PCA()
    pca.fit(reshaped)
    
    cov = pca.get_covariance()
    
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    return eigenvalues, eigenvectors


class AugmentedBatchIterator(BatchIterator):
    
    def __init__(self, batch_size, crop_size=8, testing=False):
        super(AugmentedBatchIterator, self).__init__(batch_size)
        self.crop_size = crop_size
        self.testing = testing

    def transform(self, Xb, yb):

        Xb, yb = super(AugmentedBatchIterator, self).transform(Xb, yb)
        batch_size, nchannels, width, height = Xb.shape
        
        if self.testing:
            if self.crop_size % 2 == 0:
                right = left = self.crop_size // 2
            else:
                right = self.crop_size // 2
                left = self.crop_size // 2 + 1
            X_new = Xb[:, :, right: -left, right: -left]
            return X_new, yb

        eigenvalues, eigenvectors = compute_PCA(Xb)

        # Flip half of the images horizontally at random
        indices = np.random.choice(batch_size, batch_size // 2, replace=False)        
        Xb[indices] = Xb[indices, :, :, ::-1]

        # Crop images
        X_new = np.zeros(
            (batch_size, nchannels, width - self.crop_size, height - self.crop_size),
            dtype=np.float32
        )

        for i in range(batch_size):
            # Choose x, y pixel posiitions at random
            px, py = np.random.choice(self.crop_size, size=2)
                
            sx = slice(px, px + width - self.crop_size)
            sy = slice(py, py + height - self.crop_size)
            
            # Rotate 0, 90, 180, or 270 degrees at random
            nrotate = np.random.choice(4)
            
            # add random color perturbation
            alpha = np.random.normal(loc=0.0, scale=0.5, size=5)
            noise = np.dot(eigenvectors, np.transpose(alpha * eigenvalues))
            
            for j in range(nchannels):
                X_new[i, j] = np.rot90(Xb[i, j, sx, sy] + noise[j], k=nrotate)
                
        return X_new, yb


class SaveParams(object):

    def __init__(self, name):
        self.name = name

    def __call__(self, nn, train_history):
        if train_history[-1]["valid_loss_best"]:
            nn.save_params_to("{}.params".format(self.name))
            with open("{}.history".format(self.name), "w") as f:
                pickle.dump(train_history, f)

class UpdateLearningRate(object):

    def __init__(self, start=0.001, stop=0.0001):
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.float32(self.ls[epoch - 1])
        getattr(nn, "update_learning_rate").set_value(new_value)

class TrainSplit(object):

    def __init__(self, eval_size):
        self.eval_size = eval_size

    def __call__(self, X, y, net):
        if self.eval_size:
            X_train, y_train = X[:-self.eval_size], y[:-self.eval_size]
            X_valid, y_valid = X[-self.eval_size:], y[-self.eval_size:]
        else:
            X_train, y_train = X, y
            X_valid, y_valid = _sldict(X, slice(len(y), None)), y[len(y):]

        return X_train, X_valid, y_train, y_valid

net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),

        ('conv11', layers.Conv2DLayer),
        ('conv12', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),

        ('conv21', layers.Conv2DLayer),
        ('conv22', layers.Conv2DLayer),
        ('conv23', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),

        ('conv31', layers.Conv2DLayer),
        ('conv32', layers.Conv2DLayer),
        ('conv33', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),

        ('dropout4', layers.DropoutLayer),
        ('hidden4', layers.DenseLayer),

        ('dropout5', layers.DropoutLayer),
        ('hidden5', layers.DenseLayer),

        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 5, 44, 44),
    
    conv11_num_filters=32, conv11_filter_size=(5, 5),
    conv11_nonlinearity=leaky_rectify,
    conv11_W=Orthogonal(np.sqrt(2 / (1 + 0.01**2))), conv11_b=Constant(0.1),
    
    conv12_num_filters=32, conv12_filter_size=(3, 3), conv12_pad=1,
    conv12_nonlinearity=leaky_rectify,
    conv12_W=Orthogonal(np.sqrt(2 / (1 + 0.01**2))), conv12_b=Constant(0.1),

    pool1_pool_size=(2, 2),

    conv21_num_filters=64, conv21_filter_size=(3, 3), conv21_pad=1,
    conv21_nonlinearity=leaky_rectify,
    conv21_W=Orthogonal(np.sqrt(2 / (1 + 0.01**2))), conv21_b=Constant(0.1),
    
    conv22_num_filters=64, conv22_filter_size=(3, 3), conv22_pad=1,
    conv22_nonlinearity=leaky_rectify,
    conv22_W=Orthogonal(np.sqrt(2 / (1 + 0.01**2))), conv22_b=Constant(0.1),

    conv23_num_filters=64, conv23_filter_size=(3, 3), conv23_pad=1,
    conv23_nonlinearity=leaky_rectify,
    conv23_W=Orthogonal(np.sqrt(2 / (1 + 0.01**2))), conv23_b=Constant(0.1),

    pool2_pool_size=(2, 2),

    conv31_num_filters=128, conv31_filter_size=(3, 3), conv31_pad=1,
    conv31_nonlinearity=leaky_rectify,
    conv31_W=Orthogonal(np.sqrt(2 / (1 + 0.01**2))), conv31_b=Constant(0.1),
    
    conv32_num_filters=128, conv32_filter_size=(3, 3), conv32_pad=1,
    conv32_nonlinearity=leaky_rectify,
    conv32_W=Orthogonal(np.sqrt(2 / (1 + 0.01**2))), conv32_b=Constant(0.1),
    
    conv33_num_filters=128, conv33_filter_size=(3, 3), conv33_pad=1,
    conv33_nonlinearity=leaky_rectify,
    conv33_W=Orthogonal(np.sqrt(2 / (1 + 0.01**2))), conv33_b=Constant(0.1),

    pool3_pool_size=(2, 2),

    hidden4_num_units=2048,
    hidden4_nonlinearity=leaky_rectify,
    hidden4_W=Orthogonal(np.sqrt(2 / (1 + 0.01**2))), hidden4_b=Constant(0.01),
    dropout4_p=0.5,
    
    hidden5_num_units=2048,
    hidden5_nonlinearity=leaky_rectify,
    hidden5_W=Orthogonal(np.sqrt(2 / (1 + 0.01**2))), hidden5_b=Constant(0.01),
    dropout5_p=0.5,

    output_num_units=2,
    output_nonlinearity=softmax,

    update_learning_rate=theano.shared(np.float32(0.003)),
    update_momentum=0.9,

    objective_loss_function=categorical_crossentropy,
    regression=False,
    max_epochs=750,
    batch_iterator_train=AugmentedBatchIterator(batch_size=128, crop_size=4),
    batch_iterator_test=AugmentedBatchIterator(batch_size=128, crop_size=4, testing=True),

    on_epoch_finished=[
        UpdateLearningRate(start=0.003, stop=0.0001),
        SaveParams("net")
    ],

    verbose=2,
    train_split=TrainSplit(eval_size=15000)
    )


net.fit(X, y)
            
best_valid_loss = min([row['valid_loss'] for row in net.train_history_])
print("Best valid loss: {}".format(best_valid_loss))

X_valid = X[-15000:]
y_valid = y[-15000:]

for i in range(5):
    X_valid[:, i, :, :] = renormalize(X_valid[:, i, :, :])
    
y_valid = renormalize(y_valid).astype(np.int32)

y_pred_valid = np.zeros((len(y_valid), 64))

class AugmentedBatchIterator(BatchIterator):

    def __init__(self, batch_size, crop_size=8, validation=False, testing=False, startx=None, starty=None, rotate=None):
        super(AugmentedBatchIterator, self).__init__(batch_size)
        self.crop_size = crop_size
        self.validation = validation
        self.testing = testing
        self.startx, self.starty = startx, starty
        self.rotate = rotate

    def transform(self, Xb, yb):

        Xb, yb = super(AugmentedBatchIterator, self).transform(Xb, yb)
        batch_size, nchannels, width, height = Xb.shape

        if self.validation:
            if self.crop_size % 2 == 0:
                right = left = self.crop_size // 2
            else:
                right = self.crop_size // 2
                left = self.crop_size // 2 + 1
            X_new = Xb[:, :, right: -left, right: -left]
            return X_new, yb

        if not self.testing:
            eigenvalues, eigenvectors = compute_PCA(Xb)

        # Flip half of the images horizontally at random
        indices = np.random.choice(batch_size, batch_size // 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        # Crop images
        X_new = np.zeros(
            (batch_size, nchannels, width - self.crop_size, height - self.crop_size),
            dtype=np.float32
        )

        for i in range(batch_size):
            if self.testing:
                px, py = self.startx, self.starty
            else:
                # Choose x, y pixel posiitions at random
                px, py = np.random.choice(self.crop_size, size=2)

            sx = slice(px, px + width - self.crop_size)
            sy = slice(py, py + height - self.crop_size)

            # Rotate 0, 90, 180, or 270 degrees at random
            if self.testing:
                nrotate = self.rotate
                noise = np.zeros(nchannels)
            else:
                nrotate = np.random.choice(4)
                # add random color perturbation
                alpha = np.random.normal(loc=0.0, scale=0.5, size=5)
                noise = np.dot(eigenvectors, np.transpose(alpha * eigenvalues))

            for j in range(nchannels):
                X_new[i, j] = np.rot90(Xb[i, j, sx, sy] + noise[j], k=nrotate)

        return X_new, yb

count = 0

print("Starting model combination...")

for startx in range(4):
    for starty in range(4):
        for rotate in range(4):

            net.batch_iterator_test=AugmentedBatchIterator(
                batch_size=128,
                crop_size=4,
                testing=True,
                startx=startx,
                starty=starty,
                rotate=rotate
            )
            y_pred_valid[:, count] = net.predict_proba(X_valid)[:, 1]

            count += 1

            print("Iteration: {} / 64".format(count))

combine = bmc.BMC()
combine.fit(y_pred_valid, y_valid)

print("Validation set done.")

X_test = np.load("../data/sdss_test_images.npy")
y_test = np.load("../data/sdss_test_labels.npy")

for i in range(5):
    X_test[:, i, :, :] = renormalize(X_test[:, i, :, :])

y_test = renormalize(y_test).astype(np.int32)

y_pred_test = np.zeros((len(y_test), 64))

count = 0

for startx in range(4):
    for starty in range(4):
        for rotate in range(4):

            net.batch_iterator_test=AugmentedBatchIterator(
                batch_size=128,
                crop_size=4,
                testing=True,
                startx=startx,
                starty=starty,
                rotate=rotate
            )
            y_pred_test[:, count] = net.predict_proba(X_test)[:, 1]

            count += 1

            print("Iteration: {} / 64".format(count))

y_pred = combine.predict_proba(y_pred_test)

np.save("sdss_convnet_pred.npy", y_pred)

print("Testing set done.")

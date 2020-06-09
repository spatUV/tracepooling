# tracepooling-layer
Adaptive pooling layer that applies a non-linear resampling of network activations on the temporal axis by applying a data-dependent temporal warping, which samples densely in information-rich regions and sparsely in information-poor regions.

*I. Martí-Morató, M. Cobos and F. J. Ferri, "Adaptive Distance-Based Pooling in Convolutional Neural Networks for Audio Event Classification," IEEE/ACM Transactions on Audio, Speech and Language Processing, 2020.*

TraceLayer is an adaptive (non-trainable) pooling layer which performs a non-linear temporal
transformation that follows a uniform distance subsampling criterion on the deep feature space.

The layer can be applied to any convolutional neural network model for sound event recognition, 
increasing the performance of the pre-trained model when there are mismatching test conditions.

Installation
------------

```
pip install tracepooling-layer
```


Usage
-----
TraceLayer is implemented as a [tensorflow](https://www.tensorflow.org/) layer, so it can be added in a tensorflow model as:

```
x = TraceLayer(2)(previous_layer)
```
Where the input parameter is the desired downsampling factor for the time dimension.


# tracepooling-layer
Trace pooling layer to perform pooling downsampling using trace information.

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


# Activation Functions for Neural Networks

This repository contains Python implementations of activation functions fundamental to the development of artificial neural networks. The included functions are essential components in intermediate and output layers, allowing the introduction of nonlinearities in deep learning models[1].

## Technical Description

The activation functions transform linear inputs of artificial neurons into nonlinear outputs, a critical ability for learning complex patterns. This package implements four key variants:

### ReLU (Rectified Linear Unit).
Implemented in `relu.py`, it defines the operation \(f(x) = \max(0, x)\). Its computational efficiency and mitigation of the gradient vanishing problem make it preferred in hidden layers[1]. The derivative is calculated as:

    \[
    f'(x) = \begin{cases} 
    1 & \text{if } x > 0 \.
    0 & \text{otherwise}
    \end{cases}
    \]
    
### Leaky ReLU
Present in `leaky_relu.py`, introduces a leaky parameter \(typically 0.01) to prevent dead neurons: 
    
    \[
    f(x) = \begin{cases} 
    x & \text{if } x > 0.
    \alpha x & \text{otherwise}
    \end{cases}
    \]
    
This variant maintains a minimum gradient in negative regions, improving training stability[1].
    
### Sigmoid function
In `sigmoid.py`, implements logistic compression:
    
    \[
    \sigma(x) = \frac{1}{1 + e^{-x}}
    \]
    
Ideal for output layers in binary classification because of its range (0,1). Its derivative \(\sigma'(x) = \sigma(x)(1 - \sigma(x))\) allows efficient gradient updates[1].
    
### Hyperbolic Tangent (Tanh)
In `tanh.py`, produces outputs in (-1, 1) via:
    
    \[
    \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
    \]

Preferred in hidden layers for zero-centered data, with derivative \(1 - \tanh^2(x)\)[1].

## Implementation and Code Structure

The package uses a modular architecture with unified imports in `__init__.py`:

Unified import of functions
    from .relu import relu
    from .leaky_relu import leaky_relu
    from .sigmoid import sigmoid
    from .tanh import tanh



Each function follows the standard `def function_name(x: float) -> float` signature, allowing composition with frameworks such as NumPy and TensorFlow.

## Practical Use

Example of integration in a dense neural network:
    
    import numpy as np
    from activation_functions import relu, sigmoid
    
    Hidden layer with ReLU
    hidden_layer = np.array([1.2, -0.5, 3.1])
    activated_hidden = relu(hidden_layer) # [1.2, 0.0, 3.1] # [1.2, 0.0, 3.1].
    
    Output layer with Sigmoid
    output_layer = np.array([0.8, -1.0])
    predictions = sigmoid(output_layer) # [0.68997, 0.26894] # [0.68997, 0.26894].


For Leaky ReLU with custom α:
    from activation_functions import leaky_relu.
    
    custom_leaky = leaky_relu(x=-2.5, alpha=0.1) # -0.25


## Performance Considerations

All functions are vectorized to operate on NumPy arrays. The derivative calculation is implemented by:

    def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.0).



This approach allows O(n) efficiency and optimal memory usage, critical for large datasets.


# Convolution Neural Network

CNN's are very powerfull deep learning tools. These are the the main features of CNN's which make them to prefer over [**Fully Connected Neural Networks**](https://github.com/girishdhegde/nn-lab):

1.  Spatially **local** pattern awareness.
2.  **Less parameters** due to **sparse** connections and **Parmeter sharing**.
3.  **Translational** Invariance.
4.  Capturing Global view from **receptive fields**.

 This repo. contains implementation of CNN from **scratch in pure python and numpy**.
***
## Features :
***

1.  Modular: Seperate **conv2d, maxpool2d, view, flatten layer**  classes with their own **forward** and **backward** functions.
2.  Supports **strided convolution**
3.  Supports **padding**
4.  Supports:
    
    *  Activations: Linear, **ReLU**, Sigmoid([Why we need activations](https://stackoverflow.com/a/63543274/14108734))
    
    *  Loss: MSELoss, BCELoss
    
    *  [Optimizers](https://github.com/girishdhegde/optimizers): SGD, Momentum SGD, RMSprop, **Adam**
To Do
5.  Generalization: CNN's of any custom shape should be created
6.  Support **Visualization**:
 
## Here's How To Run The Code:
***
### Requirements:
1.  numpy
2.  matplotlib(For visualization if required)


# To Do:
Only **layers** are implemented. So need to join these layers to create fully functional CNN and should support **Fully Connected** output layers.

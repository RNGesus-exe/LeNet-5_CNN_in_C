# LeNet-5_CNN_in_C
I have implemented the LeNet-5 CNN in C.

## LeNet-5 CNN Architecture
    1. The input layer image is 1x1x28x28
    2. The first hidden layer is a convolution layer [p=2,s=1,f=5,a=tanh] which outputs 1x1x28x28
    3. The second layer is a subsampling [p=0,s=2,f=2,a=avg_pool] which outputs 1x6x14x14
    4. The third layer is a convolution [p=0,s=1,f=5,a=tanh] which outputs 1x6x10x10
    5. The fourth layer is a subsampling [p=0,s=2,f=2,a=avg_pool] which outputs 1x16x5x5
    6. The fifth layer is a convolution [p=0,s=1,f=5,a=tanh] which outputs 1x120x1x1
    7. We will flatten the 1x120x1x1 feature maps into 1x120
    8. The sixth layer is fully connected [a=tanh] which outputs 1x84x1x1
    9. The seventh layer is a fully connected [a=softmax] which outputs 1x10x1x1
    10. The output layer will have an argmax() to tell us which neuron from seventh layer had highest probability

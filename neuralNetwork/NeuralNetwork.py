import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))


from dataclasses import dataclass
import numpy as np
from tensorflow.keras.losses import BinaryCrossentropy


@dataclass
class Activation:
    activation:str='sigmoid'

    def __post_init__(self):
        self.activation_function= self.ReLU
        self.activation_function_prime= self.ReLU_prime


    def tanh(self,x):
            return np.tanh(x)

    def tanh_prime(self,x):
        return 1 - np.tanh(x) ** 2

    def sigmoid(self,Z:np.ndarray):
        return np.power(1+np.exp(-np.longdouble(Z)),-1)

    def sigmoid_prime(self,Z):
        return np.dot(self.sigmoid(np.longdouble(Z)),1-self.sigmoid(np.longdouble(Z)))

    def ReLU(self,Z):
        return np.maximum(0,Z)
    def ReLU_prime(self,Z):
        return  1*(Z>0)


    def forward_prop(self,X):
        self.input=X
        return self.activation_function(X)

    def back_prop(self,output):
        p=self.activation_function_prime(self.input)
        return np.dot(output,p).T



@dataclass
class _Layer:
    name:str
    units:int
    activation:str='sigmoid'
    lam:float=20

    def __post_init__(self):
        self.weights=None
        self.biases=None
        self.activation_instance=Activation(self.activation)

    def forward_prop(self,X:np.ndarray):
        self.num_of_samples,self.num_of_feat=X.shape
        if self.weights is None and self.biases is None:
            self.weights=np.random.rand(self.units,self.num_of_feat)
            self.biases=np.random.rand(self.units,1)
        self.input=X
        # column vector
        col=np.dot(self.weights,X.T)
        return self.activation_instance.forward_prop((col + self.biases).T)

    def back_prop(self,output:np.ndarray,lr:float):
        weights_gradient=np.dot(output.T,self.input)
        input=np.dot(self.weights.T,output.T)
        self.weights=self.weights*(1-(lr * self.lam/self.num_of_samples))-  lr*weights_gradient
        b=np.sum(lr*output,axis=0).reshape(-1,1)
        self.biases-=b
        return self.activation_instance.back_prop(input)

@dataclass
class NeuralNetwork:
    layers:list[tuple[str,int]]
    epochs:int=1000
    lr:float=0.01

    def __post_init__(self):
        self.layers=[_Layer(name,units) for name,units in self.layers]
        # self.layers[len(self.layers)-1].activation_func=Activation().sigmoid

    def get_weights(self):
        for l in self.layers:
            print('='*20)
            print(l.name)
            print(f'Weights>>>')
            print(l.weights)
            print(f'Biases>>>')
            print(l.biases)

    def forward_prop(self,X):
        output=X
        layer:_Layer
        for layer in self.layers:
            output=layer.forward_prop(output)
        return output


    def binary_cross_entropy(self,y_true, y_pred):
        # res=None
        # try:
        #     res=np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))
        # except:
        #     print(y_true,y_pred)
        #     return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))
        # return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))
        return BinaryCrossentropy()(y_true,y_pred).numpy()

    def binary_cross_entropy_prime(self,y_true, y_pred):
        return ((1 - y_true) / (1 - y_pred) - y_true / (y_pred)) / np.size(y_true)



    def fit(self,X,Y):
        for e in range(self.epochs):
            error=0
            output=self.forward_prop(X)
            # output=self.sigmoid(output)

            error+=self.binary_cross_entropy(Y,output)

            error_grad=self.binary_cross_entropy_prime(Y,output)
            layer:_Layer
            for layer in self.layers[::-1]:
                error_grad=layer.back_prop(error_grad,lr=self.lr)

            error/=X.shape[0]
            if not (e%100):
                print(f'epoch>>>f{e/100}')

    def sigmoid(self,Z:np.ndarray):
        return np.power(1+np.exp(-np.longdouble(Z)),-1)


    def predict(self,X):
        return (self.forward_prop(X)>=0.5).astype(int)

a=np.array([[1,2,5],[2,3,6]])
b=np.array([[1,2],[3,5],[6,7]])

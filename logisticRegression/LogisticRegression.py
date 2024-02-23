
from dataclasses import dataclass
import numpy as np



@dataclass
class LogisticRegression:
    weights:np.ndarray=None
    lr:int=0.001
    num_iters:int=int(1e3)
    threshold:float=0.5
    bias:float=0

    def fit(self,X:np.ndarray,Y:np.ndarray)->None:
        self.num_of_samples,self.num_of_feat=X.shape
        if  self.weights is None:
            self.weights=np.zeros(self.num_of_feat)

        for _ in range(self.num_iters):
            # w.x
            wx=np.dot(X,self.weights)
            # w.x + b
            wxb=wx+self.bias
            # calculate the sigmoid for each data point f(x)
            f=self.__sigmoid(wxb)
            # f(x)-y
            diff=f-Y
            for index,_ in enumerate(self.weights):
                self.weights[index]-=self.lr*np.dot(diff,X[:,index])/self.num_of_samples
            self.bias-=self.lr * np.sum(diff) / self.num_of_samples


    def predict(self,X:np.ndarray)->np.ndarray:
        res=self.__sigmoid(np.dot(X,self.weights) + self.bias)
        return (self.threshold<=res).astype(int)


    def set_weights(self,W:np.ndarray)->None:
        self.weights=W

    def get_weights(self)->np.ndarray:
        return self.weights


    def __sigmoid(self,Z:np.ndarray)->np.ndarray:
        return np.power((1+np.exp(-1*Z)),-1)
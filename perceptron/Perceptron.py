from dataclasses import dataclass
import numpy as np

@dataclass
class Perceptron:
    lr:float=0.01
    weights:np.ndarray=None
    bias:float=0
    num_iters:int=1000


    def fit(self,X:np.ndarray,Y:np.ndarray)->None:
        self.num_of_samples,self.num_of_feat=X.shape
        self.weights=np.zeros(self.num_of_feat)
        # ensure the Y values to be either 1 or 0
        y_=(Y>0).astype(int)
        for _ in range(self.num_iters):
            z=np.dot(X,self.weights)+self.bias
            preds=self.g(z)
            preds=y_-preds
            dw=self.lr*np.dot(X.T,preds)
            self.weights+=dw
            self.bias+=self.lr*np.sum(preds)

    def predict(self,X:np.ndarray)->np.ndarray:
        return self.g(np.dot(X,self.weights)+self.bias)


    # unit step function
    def g(self,Z:np.ndarray)->np.ndarray:
        return (Z>=0).astype(int)

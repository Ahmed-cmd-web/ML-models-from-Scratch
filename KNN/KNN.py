from dataclasses import dataclass
import numpy as np




@dataclass
class KNN:
    K:int=3
    type:str='R'

    def fit(self,X:np.ndarray,Y:np.ndarray)->None:
        self.X_train=X
        self.Y_train=Y


    def predict(self,X:np.ndarray)->np.ndarray:

        computed_distances:np.ndarray=np.apply_along_axis(lambda t:np.sum((t-self.X_train)**2,axis=1)**1/2,1,X)
        indices=np.argsort(computed_distances)
        indices=indices[:,:self.K]
        if self.type=='R':
            return np.apply_along_axis(lambda indicesRow:np.average(self.Y_train[indicesRow]),1,indices)
        return np.apply_along_axis(lambda indicesRow:np.argmax(np.bincount(self.Y_train[indicesRow])),1,indices)



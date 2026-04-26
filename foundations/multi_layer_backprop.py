import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x=np.array(x).reshape(1, -1)
        W1=np.array(W1)
        b1=np.array(b1).reshape(1, -1)
        W2=np.array(W2)
        b2=np.array(b2).reshape(1, -1)
        y_true=np.array(y_true).reshape(1, -1)
        # forward pass
        z1=np.dot(x, W1.T)+b1
        a1=np.maximum(0, z1) # ReLU
        z2=np.dot(a1, W2.T)+b2
        loss=np.mean((z2-y_true)**2)
        # backward pass
        n=len(y_true) if y_true.ndim>0 else 1
        dz2=(2/n)*(z2-y_true)
        dW2=np.dot(dz2.T, a1)
        db2=np.sum(dz2, axis=0)
        # gradient
        da1=np.dot(dz2, W2)
        dz1=da1.copy()
        dz1[z1<=0]=0
        dW1=np.dot(dz1.T, x)
        db1=np.sum(dz1, axis=0)
        return {
            'loss': np.round(float(loss), 4),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist()
        }
        pass

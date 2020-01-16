import numpy as np

    
class SGD:
    '''
    (Stochastic Gradient Descent）
    '''
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for i in range(len(params)):
            
            params[i] -= self.lr * grads[i]

    def _zero_grad(self, grads):
        for grad in grads:
            grad[...] = 0
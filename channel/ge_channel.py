from .base_channel import *

class GE_Channel(Channel):
    def __init__(self, alpha, beta, epsilon, delta = 1):
        self.alpha = alpha
        self.beta = beta
        self.eps = epsilon
        self.delta = delta

    def model(self, num_packets):
        isBadState = False
        drop_mask = np.zeros(num_packets, dtype=bool)
        
        for i in range(num_packets):
            if not isBadState:
                if np.random.rand() < self.eps: 
                    drop_mask[i] = True  # isolated loss (epsilon)
                if np.random.rand() < self.alpha:
                    isBadState = True  
            else: 
                if np.random.rand() < self.delta:
                    drop_mask[i] = True # burst loss when delta = 1
                if np.random.rand() < self.beta: # end burst loss
                    isBadState = False
        return drop_mask
    
    def apply(self, data: 'CodewordBatch'):
        return super().apply(data)

        
        
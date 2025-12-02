from .base_channel import *

class GE_Channel(Channel):
    def __init__(self, alpha, beta, epsilon, delta = 1):
        self.alpha = alpha
        self.beta = beta
        self.eps = epsilon
        self.delta = delta

    def simulate_loss_rates(self, num_packets: int, num_trials: int, chunk_size: int = 100000):
        """
        Run the channel model for many trials and return the average loss rate per trial.
        Trials are processed in chunks to keep the temporary arrays small.
        """
        chunk_size = max(1, min(chunk_size, num_trials))
        loss_rates = np.empty(num_trials, dtype=np.float64)
        start = 0
        while start < num_trials:
            end = min(start + chunk_size, num_trials)
            loss_rates[start:end] = self._simulate_chunk(num_packets, end - start)
            start = end
        return loss_rates

    def _simulate_chunk(self, num_packets: int, chunk_trials: int):
        is_bad = np.zeros(chunk_trials, dtype=bool)
        loss_counts = np.zeros(chunk_trials, dtype=np.int32)

        for _ in range(num_packets):
            isolated_losses = (~is_bad) & (np.random.rand(chunk_trials) < self.eps)
            loss_counts += isolated_losses.astype(np.int32)

            enter_bad = (~is_bad) & (np.random.rand(chunk_trials) < self.alpha)

            burst_losses = is_bad & (np.random.rand(chunk_trials) < self.delta)
            loss_counts += burst_losses.astype(np.int32)

            exit_bad = is_bad & (np.random.rand(chunk_trials) < self.beta)
            is_bad = (is_bad | enter_bad) & (~exit_bad)

        return loss_counts.astype(np.float64) / num_packets

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

        
        

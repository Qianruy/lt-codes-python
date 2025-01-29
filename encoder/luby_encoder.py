from .base_encoder import *

class LubyEncoder(Encoder):
    """
    Luby Transform Encoder
    @field(data): all the inputs, array of shape [l]
    @field(prob): cummulative sum of degree distribution probability
    """
    def __init__(self, dd: np.ndarray, block: int, seed: int = 42):
        """
        @param(dd): degree distribution array of shape [d]
        @param(block): the input code word number
        """
        super().__init__()
        self.data = np.zeros((1, block), dtype=np.uint8)
        self.prob = dd.cumsum(0); self.prob[-1] = 1
        self.rng = np.random.default_rng(seed=seed)

    def get_one(self) -> Codeword:
        """
        @return sample a degree d. then xor d inputs into a codeword
        """
        degree = (self.rng.random() > self.prob).sum()
        indices = self.rng.choice(np.arange(1, self.data.shape[0]), (self.prob.shape[0],), replace=False)
        indices = (np.arange(1, self.prob.shape[0] + 1) <= degree) * indices
        # indices = self.rng.choice(np.arange(1, self.data.shape[0]), (degree,), replace=False)
        data = np.bitwise_xor.reduce(self.data[indices])
        return Codeword(indices, data, degree)

    def get_bat(self, batch: int) -> CodewordBatch:
        """
        @param(batch) the size of the batch
        @return sample multiple degrees [..d], for each [..d], xor d inputs into a codeword
        """
        degrees = (self.rng.random(size=(batch, 1)) > self.prob).sum(axis=-1) 
        indices = np.zeros((batch, self.prob.shape[0]), dtype=np.int_)
        seeds = self.rng.random(size=batch)
        @delayed
        def fill(b):
            rng = np.random.default_rng(int(10000 * seeds[b]))
            # print(self.data.shape[0], self.prob.shape[0])
            indices[b] = rng.choice(np.arange(1, self.data.shape[0]), (self.prob.shape[0],), replace=False)
        Parallel(n_jobs=4, require='sharedmem')(fill(b) for b in range(batch))
        indices = (np.arange(1, self.prob.shape[0] + 1) <= degrees.reshape(-1, 1)) * indices
        data = np.bitwise_xor.reduce(self.data[indices], axis=1)
        return CodewordBatch(indices, data, degrees)

    def put_one(self, data: np.ndarray):
        """
        @param(data) one input packet
        """
        assert data.dtype == np.uint8
        assert len(data.shape) == 1
        assert data.shape[-1] == self.data.shape[-1]
        self.data = np.concatenate([self.data, data.reshape(1, -1)], axis=0)

    def put_bat(self, data: np.ndarray):
        """
        @param(data) a batch of input packets 
        """
        assert data.dtype == np.uint8
        assert len(data.shape) == 2
        assert data.shape[-1] == self.data.shape[-1]
        self.data = np.concatenate([self.data, data], axis=0)

    def random_degree(dist_name, N, k):
        """
        @param(dist_name) selected distribution
        @param(N) maximum degree
        @param(k) number of codewords
        """
        if dist_name == "ideal":
            probabilities = ideal_distribution(N)
        elif dist_name == "robust":
            probabilities = robust_distribution(N)
        else:
            probabilities = None
        
        population = list(range(0, N+1))
        # the degree of the first drop is 1 to ensure the start of decoding
        return [1] + choices(population, probabilities, k=k-1)

if __name__ == '__main__':
    encoder = LubyEncoder(np.array(ideal_distribution(10)), 1024)
    encoder.put_bat(np.zeros((100, 1024), dtype=np.uint8))
    # encoder.put_one(np.zeros(1024, dtype=np.uint8))
    print(encoder.get_one())
    print(encoder.get_bat(3))
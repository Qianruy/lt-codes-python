from .luby_encoder import *

class SlidingFountainEncoder(LubyEncoder):
    def __init__(self, dd, block, wdn_size: int= 2000, overlap = 0.5, seed = 42):
        super().__init__(dd, block, seed)
        self.wdn_size = wdn_size
        self.overlap = overlap
        self.start = 0

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
            indices[b] = rng.choice(np.arange(self.start+1, self.start+self.wdn_size+1), (self.prob.shape[0],), replace=False)
        Parallel(n_jobs=4, require='sharedmem')(fill(b) for b in range(batch))
        indices = (np.arange(1, self.wdn_size+1) <= degrees.reshape(-1, 1)) * indices
        # Dealing with indices because some symbols are removed after shifting
        selected_indices = np.where(indices > self.start, indices-self.start, indices)
        data = np.bitwise_xor.reduce(self.data[selected_indices], axis=1)
        return CodewordBatch(indices, data, degrees)


    def shift_window(self):
        self.start += int((1- self.overlap) * self.wdn_size)
        # print("start: {}".format(self.start))

    def remove_one(self):
        """
        remove one source from encoder
        """
        pass

    def remove_bat(self, batch: int):
        """
        remove batch sources from encoder
        """
        assert(batch <= self.data.shape[0])
        self.data = np.concatenate([self.data[0:1], self.data[batch+1:]])
    

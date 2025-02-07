from .base_encoder import *
import time 
from numba import *

class WalzerEncoder(Encoder):
    """
    Walzer Encoder for real time streaming
    """
    def __init__(self, block: int, wdn_size: int = 200, redundancy: int = 1.2, 
                 mindegree: int = 3, maxdegree: int = 5, seed: int = 42, mode = 1):
        """
        """
        super().__init__()
        self.wdn = wdn_size; self.redundancy = redundancy
        self.mindegree = mindegree; self.maxdegree = maxdegree
        self.data = np.zeros((1, block), dtype=np.uint8)
        self.rng = np.random.default_rng(seed=seed)
        self.mode = mode

    def get_one(self) -> Codeword:
        raise NotImplementedError("Only support full encoding")

    def get_bat(self, batch: int) -> CodewordBatch:
        raise NotImplementedError("Only support full encoding")

    def get_all(self) -> CodewordBatch:
        """
        get codeword from encoder
        """
        batch = math.ceil(self.data.shape[0] * self.redundancy)
        degrees = np.zeros(batch, dtype=np.int8)
        seeds = self.rng.integers(0, int(1e6), size=batch)
        indices = np.zeros((batch, self.maxdegree*5), dtype=np.int32)

        def fill(b):
            # generate edges for each source symbol to connect the codewords
            rng = np.random.default_rng(seed=seeds[b])
            encode_range = int(self.wdn * self.redundancy)

            # Mode 1: original uniform dist, mode 2: add a determinist 1st edge
            if self.mode == 1:
                selected = rng.choice(encode_range, size=self.maxdegree-1, replace=False)
                selected = list(selected)+[0]
            elif self.mode == 2:
                elected = rng.choice(encode_range, size=self.maxdegree, replace=False)
            selected.sort()

            # Mode 3: Stretch the randomly and uniformly selected indexes to 
            # make the last index equal to the encode range
            if self.mode == 3:
                scaling =  encode_range / selected[-1]
                selected = np.array(selected)
                selected = (encode_range - (selected * scaling)).tolist()
                selected.pop()
                selected.append(math.ceil(b * self.redundancy)) # add 1st determinist connection
            
            # unstretched version
            selected = [x + int(b*self.redundancy) for x in list(selected)]
            for selected_index in selected:
                selected_index = int(selected_index)
                if selected_index >= batch: continue
                indices[selected_index][degrees[selected_index]] = b
                degrees[selected_index] += 1

            # if b < 100: print(b, selected_stretched)
        Parallel(n_jobs=4, require='sharedmem')(delayed(fill)(b) for b in range(1, self.data.shape[0]))
        print("Maximum degree number of the codewords: {}".format(degrees.max()))
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

if __name__ == '__main__':
    encoder = WalzerEncoder(1024, wdn_size=128, redundancy=1.05, maxdegree=4)
    encoder.put_bat(np.zeros((50000, 1024), dtype=np.uint8))
    begin = time.time() 
    encoder.get_all()
    end = time.time() 
    print(f"Runtime of encoding is {end - begin}") 
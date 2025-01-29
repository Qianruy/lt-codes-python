from .base_encoder import *
import time 
from numba import *

class PlowEncoder(Encoder):
    """
    Plow Encoder for real time streaming
    @field(ring) the ring buffer for all inputs
    """
    def __init__(self, block: int, wdn_size: int = 200, redundancy: int = 1.2, 
                 mindegree: int = 3, maxdegree: int = 5, seed: int = 42):
        """
        """
        super().__init__()
        self.wdn = wdn_size; self.redundancy = redundancy
        self.mindegree = mindegree; self.maxdegree = maxdegree
        self.data = np.zeros((1, block), dtype=np.uint8)
        self.rng = np.random.default_rng(seed=seed)

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
            codeword_idx = math.ceil(b * self.redundancy)
            selected = [math.ceil(b * self.redundancy)]
            indices[codeword_idx][degrees[codeword_idx]] = b
            degrees[codeword_idx] += 1
            encode_range = int(self.wdn * self.redundancy)

            for k in range(2, self.maxdegree + 1):
                # Generate a random value between (1 - 1/(k-1)) and (1 - 1/k)
                # upper_bound = 1 - 1 / k 
                base = math.exp(1) - 1
                upper_bound = 1 - 1/pow(base,k-1)

                # Using binomial random generation
                random_point = rng.binomial(encode_range - 1, upper_bound)

                # Calculate the index based on the random point
                selected_index = max(0, int(b * self.redundancy + random_point + 1))
                selected.append(selected_index)
                if selected_index >= batch: continue
                indices[selected_index][degrees[selected_index]] = b
                degrees[selected_index] += 1
            # if b < 100: print(b, selected)
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

    def generate_degrees(self, batch: int, ratio: int = 0.8):
        """
        Generate a list of degrees for a batch based on a weighted ratio.

        Args:
            batch (int): The total number of degrees to generate.
            ratio (float): The percentage of max_degree in the batch (default: 80%).

        Returns:
            np.ndarray: An array of degrees with the specified distribution.
        """
        num_min = int(batch * ratio); num_max = batch - num_min
        degrees = np.concatenate([
            np.full(num_min, self.mindegree, dtype=np.int_),
            np.full(num_max, self.maxdegree, dtype=np.int_)
        ])
        self.rng.shuffle(degrees)
        return degrees

if __name__ == '__main__':
    encoder = PlowEncoder(1024, wdn_size=128, redundancy=1.05, maxdegree=4)
    encoder.put_bat(np.zeros((500000, 1024), dtype=np.uint8))
    begin = time.time() 
    encoder.get_bat(200000)
    end = time.time() 
    print(f"Runtime of encoding is {end - begin}") 
from .base_encoder import *
import time 
from numba import *

class PlowEncoder(Encoder):
    """
    Plow Encoder for real time streaming
    @field(ring) the ring buffer for all inputs
    """
    def __init__(self, block: int, wdn_size: int = 200, redundancy: int = 1.2, 
                 mindegree: int = 3, maxdegree: int = 5, seed: int = 42, 
                 tail_reduction: bool = False, mode: int = 1):
        """
        """
        super().__init__()
        self.wdn = wdn_size; self.redundancy = redundancy
        self.mindegree = mindegree; self.maxdegree = maxdegree
        self.data = np.zeros((1, block), dtype=np.uint8)
        self.rng = np.random.default_rng(seed=seed)
        self.has_tail = tail_reduction
        self.mode = mode

    def get_one(self) -> Codeword:
        raise NotImplementedError("Only support full encoding")

    def get_bat(self, batch: int) -> CodewordBatch:
        raise NotImplementedError("Only support full encoding")

    def get_all(self) -> CodewordBatch:
        """
        get codeword from encoder
        """
        tail = 0; batch = math.ceil(self.data.shape[0] * self.redundancy)
        if self.has_tail: tail = math.ceil(self.wdn * self.redundancy) # add tail in the end
        seqno = np.arange(1, 1+batch+tail, dtype=np.int32)
        degrees = np.zeros(batch+tail, dtype=np.int32)
        seeds = self.rng.integers(0, int(1e6), size=batch+tail)
        indices = np.zeros((batch+tail, self.maxdegree*5), dtype=np.int32)

        def fill(b):
            # generate edges for each source symbol to connect the codewords
            rng = np.random.default_rng(seed=seeds[b])
            codeword_idx = math.ceil(b * self.redundancy)
            selected = [math.ceil(b * self.redundancy)]
            if self.mode == 1: selected = [] # Mode 1: without the 1st determinist edge
            selected_index = -1
            indices[codeword_idx][degrees[codeword_idx]] = b
            degrees[codeword_idx] += 1
            encode_range = int(self.wdn * self.redundancy)

            for k in range(2, self.maxdegree + 1):
                # Generate a random value between (1 - 1/(k-1)) and (1 - 1/k)
                # upper_bound = 1 - 1 / k 
                base = 2
                upper_bound = 1 - 1/pow(base,k-1)

                # Using binomial random generation
                random_point = rng.binomial(encode_range - 1, upper_bound)

                # Calculate the index based on the random point
                # while selected_index == -1 or selected_index in selected:
                selected_index = max(0, int(b * self.redundancy + random_point + 1))
                # Add the exception handler
                if selected_index in selected:
                    random_point = rng.binomial(encode_range - 1, upper_bound)
                    selected_index = max(0, int(b * self.redundancy + random_point + 1))
                assert(selected_index not in selected)
                selected.append(selected_index)
                if selected_index >= batch:
                    if not self.has_tail: continue
                    # tail reduction
                    else: 
                        assert(batch >= codeword_idx)
                        scaling = (codeword_idx - batch + tail) / (2 * tail)
                        selected_index = max(0, int(b * self.redundancy + random_point * scaling + 1)) 
                indices[selected_index][degrees[selected_index]] = b
                degrees[selected_index] += 1
            # if b < 100: print(b, selected)
        Parallel(n_jobs=4, require='sharedmem')(delayed(fill)(b) for b in range(1, self.data.shape[0]))
        print("Maximum degree number of the codewords: {}".format(degrees.max()))
        data = np.bitwise_xor.reduce(self.data[indices], axis=1)
        return CodewordBatch.from_dense(seqno, indices, degrees, data)


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

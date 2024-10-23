from tools import *
from distributions import *

class Encoder(ABC):
    @abstractmethod
    def get_one(self) -> Codeword:
        """
        get codeword from encoder
        """
        pass

    @abstractmethod
    def get_bat(self, batch: int) -> CodewordBatch:
        """
        get codeword from encoder
        """
        pass

    @abstractmethod
    def put_one(self, data: np.ndarray):
        """
        put input into encoder
        """
        pass

    @abstractmethod
    def put_bat(self, data: np.ndarray):
        """
        put input into encoder
        """
        pass

class LubyEncoder(Encoder):
    """
    Luby Transform Encoder
    @field(data): all the inputs, array of shape [l]
    @field(prob): cummulative sum of degree distribution probability
    """
    def __init__(self, dd: np.ndarray, psize: int):
        """
        @param(dd): degree distribution array of shape [d]
        @param(psize): the input code word size
        """
        super().__init__()
        self.data = np.zeros((1, psize), dtype=np.uint8)
        self.prob = dd.cumsum(0)
        self.prob[-1] = 1

    def get_one(self) -> Codeword:
        """
        @return sample a degree d, and xor d inputs into a codeword
        """
        degree  = (np.random.random() > self.prob).sum()
        index   = np.random.randint(1, self.data.shape[0], size=(degree,))
        data    = np.bitwise_xor.reduce(self.data[index])
        return Codeword(index, data, degree)

    def get_bat(self, batch: int) -> CodewordBatch:
        """
        @param(batch) the size of the batch
        @return sample multiple degrees [..d], for each [..d], xor d inputs into a codeword
        """
        degree  = (np.random.random(size=(batch, 1)) > self.prob).sum(axis=-1) + 1
        index   = np.random.randint(1, self.data.shape[0], size=(batch, degree.max(),))
        index   = (np.arange(1, degree.max() + 1) <= degree.reshape(batch, 1)) * index
        data    = np.bitwise_xor.reduce(self.data[index])
        return CodewordBatch(index, data, degree)

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

class PlowEncoder(Encoder):
    """
    Plow Encoder for real time streaming
    @field(ring) the ring buffer for all data packets
    @field(head) the head pointer for ring buffer
    @field(tail) the tail pointer for ring buffer
    """
    def __init__(self, rsize: int):
        """
        @param(rsize): the maximum size of ring buffer
        """
        self.ring = np.zeros()
        self.head = 0
        self.tail = 0

if __name__ == '__main__':
    encoder = LubyEncoder(np.array([0.5, 0.5]), 1024)
    encoder.put_one(np.zeros(1024, dtype=np.uint8))
    encoder.put_bat(np.ones((100, 1024), dtype=np.uint8))
    print(encoder.get_bat(7))
    print(encoder.get_one())

WINDOWSIZE = 300
min_degree = 5
max_degree = 5

def get_degrees_from(distribution_name, N, k):
    """ Returns the random degrees from a given distribution of probabilities.
    The degrees distribution must look like a Poisson distribution and the 
    degree of the first drop is 1 to ensure the start of decoding.
    """

    if distribution_name == "ideal":
        probabilities = ideal_distribution(N)
    elif distribution_name == "robust":
        probabilities = robust_distribution(N)
    else:
        probabilities = None
    
    population = list(range(0, N+1))
    return [1] + choices(population, probabilities, k=k-1)
   
def encode(blocks, redundancy, codetype):
    """ Iterative encoding - Encodes new symbols and yield them.
    Encoding one symbol is described as follow:

    1.  Randomly choose a degree according to the degree distribution, save it into "deg"
        Note: below we prefer to randomly choose all the degrees at once for our symbols.

    2.  Choose uniformly at random 'deg' distinct input blocs. 
        These blocs are also called "neighbors" in graph theory.
    
    3.  Compute the output symbol as the combination of the neighbors.
        In other means, we XOR the chosen blocs to produce the symbol.
    """

    # Display statistics
    blocks_n = len(blocks)
    drops_quantity = int(blocks_n * redundancy)
    assert blocks_n <= drops_quantity, "Because of the unicity in the random neighbors, it is need to drop at least the same amount of blocks"

    print(f"WINDOWSIZE = {WINDOWSIZE}")
    print("Generating graph...")
    start_time = time.time()

    print("Ready for encoding.", flush=True)

    if codetype == "LT":
        # Generate random indexes associated to random degrees, seeded with the symbol id
        random_degrees = get_degrees_from("robust", blocks_n, k=drops_quantity)

        for i in range(drops_quantity):
            
            # Get the random selection, generated precedently (for performance)
            selection_indexes, deg = generate_indexes(i, random_degrees[i], 0, blocks_n)
            # Xor each selected array within each other gives the drop (or just take one block if there is only one selected)
            drop = blocks[selection_indexes[0]]
            for n in range(1, deg):
                drop = np.bitwise_xor(drop, blocks[selection_indexes[n]])
                # drop = drop ^ blocks[selection_indexes[n]] # according to my tests, this has the same performance

            # Create symbol, then log the process
            symbol = Symbol(index=i, degree=deg, data=drop, neighbors=selection_indexes)
            
            if VERBOSE:
                symbol.log(blocks_n)

            log("Encoding", i, drops_quantity, start_time)

            yield symbol
    elif codetype == "PLOW":

        # preparation before start encoding
        encode_range = int(WINDOWSIZE*redundancy)
        symbols = [Symbol(index=i) for i in range(WINDOWSIZE)]

        for i in range(blocks_n):

            deg = random.randint(min_degree, max_degree) # some symbols will be empty
            # selection_indexes = [i*redundancy] + [i*redundancy + int(encode_range*(k-1)/k)-1 for k in range(2,deg+1)]
            selection_indexes = generate_indexes_plow(i, redundancy, encode_range, deg)
            print(len(symbols), selection_indexes)

            # Xor the input block to each selected encoded symbol
            for idx in selection_indexes:
                
                while idx >= len(symbols): 
                    symbol = Symbol(index=len(symbols)+1)
                    symbols.append(symbol)

                if symbols[idx].degree == 0: symbols[idx].data = blocks[i]
                else: symbols[idx].data = np.bitwise_xor(symbols[idx].data, blocks[i])

                # Update symbol attributes
                symbols[idx].degree += 1
                symbols[idx].neighbors.add(i)

            # Create a new symbol to the end of symbol list
            # symbols.append(Symbol(index=len(symbols)+1))

            # if not i % 3000:
            #     symbols = [Symbol(index=len(symbols)+1, data=blocks[i]) for i in range(WINDOWSIZE)]
            
            log("Encoding", i, blocks_n, start_time)
        
        for symbol in symbols: yield symbol

    print("\n----- Correctly dropped {} symbols (packet size={})".format(drops_quantity, PACKET_SIZE))

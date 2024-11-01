from encoder import *
from decoder import *

if __name__ == '__main__':
    encoder = LubyEncoder(np.array([0.5, 0.25, 0.25]), 1024, 10000)
    encoder.put_bat(np.ones((1000, 1024), dtype=np.uint8))
    decoder = LubyDecoder(3, 1024)
    for i in range(5):
        decoder.put_bat(encoder.get_bat(10000))
    print((decoder.get_all() == np.ones((1000, 1024), dtype=np.uint8)).all())
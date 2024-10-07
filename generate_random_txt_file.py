import random
import string
import argparse

def generate_random_text_file(filename, filesize):
    chars = string.ascii_letters + string.digits + string.punctuation + ' '

    with open(filename, 'w') as f:
        size_written = 0
        while size_written < filesize:
            chunk_size = min(1024, filesize - size_written)
            random_text = ''.join(random.choice(chars) for _ in range(chunk_size))
            f.write(random_text)
            size_written += chunk_size

    print(f"{filename} successfully generated, size = {filesize} bytes.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='file name')
    parser.add_argument('filesize', type=int, help='file size')

    args = parser.parse_args()

    generate_random_text_file(args.filename, args.filesize)
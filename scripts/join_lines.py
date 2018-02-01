#!/usr/bin/env python3
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--n-chars', type=int, default=1000)
    args = parser.parse_args()

    with open(args.filename) as f:
        buff = []
        for line in f:
            buff.append(line.strip())
            if sum(map(len, buff)) >= args.n_chars:
                print(' '.join(buff))
                buff = []
        print(' '.join(buff))


if __name__ == '__main__':
    main()

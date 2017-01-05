#!/usr/bin/env python3

import argparse
import os.path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus')
    parser.add_argument('--lines', type=int, default=300000)
    args = parser.parse_args()

    n_files = 0
    n_lines = 0
    outf = None
    with open(args.corpus) as f:
        for line in f:
            if outf is None:
                if '.' in os.path.basename(args.corpus):
                    corpus_name, ext = args.corpus.rsplit('.', 1)
                else:
                    corpus_name, ext = args.corpus, 'txt'
                name = '{}-{}.{}'.format(
                    corpus_name, str(n_files).zfill(4), ext)
                print('Starting {}'.format(name))
                outf = open(name, 'w')
                n_files += 1
            outf.write(line)
            n_lines += 1
            if n_lines >= args.lines:
                outf.close()
                outf = None
                n_lines = 0


if __name__ == '__main__':
    main()

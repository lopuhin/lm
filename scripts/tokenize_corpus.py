#!/usr/bin/env python3

import argparse
import re

from nltk.tokenize import wordpunct_tokenize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus')
    parser.add_argument('output')
    parser.add_argument('--no-punct', action='store_true')
    parser.add_argument('--min-length', type=int)
    args = parser.parse_args()

    if args.no_punct:
        tokenize = lambda s: re.findall('\w+', s)
    else:
        tokenize = wordpunct_tokenize

    with open(args.corpus) as f:
        with open(args.output, 'w') as outf:
            for line in f:
                tokens = tokenize(line.strip())
                if not args.min_length or len(tokens) >= args.min_length:
                    outf.write(' '.join(tokens).lower())
                    outf.write('\n')


if __name__ == '__main__':
    main()

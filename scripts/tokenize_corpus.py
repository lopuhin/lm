#!/usr/bin/env python3

import argparse
import re

from nltk.tokenize import wordpunct_tokenize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus')
    parser.add_argument('output')
    parser.add_argument('--no-punct', action='store_true')
    args = parser.parse_args()

    if args.no_punct:
        tokenize = lambda s: re.findall('\w+', s)
    else:
        tokenize = wordpunct_tokenize

    with open(args.corpus) as f:
        with open(args.output, 'w') as outf:
            for line in f:
                line = line.strip()
                outf.write(' '.join(tokenize(line)).lower())
                outf.write('\n')


if __name__ == '__main__':
    main()

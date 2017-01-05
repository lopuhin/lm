#!/usr/bin/env python3

import argparse
import re

from nltk.tokenize import wordpunct_tokenize, sent_tokenize


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('corpus')
    arg('output')
    arg('--no-punct', action='store_true')
    arg('--min-length', type=int)
    arg('--sent-tokenize', action='store_true',
        help='assume sentences do not span lines')
    args = parser.parse_args()

    if args.no_punct:
        tokenize = lambda s: re.findall('\w+', s)
    else:
        tokenize = wordpunct_tokenize

    with open(args.corpus) as f:
        with open(args.output, 'w') as outf:
            for line in f:
                if args.sent_tokenize:
                    sentences = sent_tokenize(line)
                else:
                    sentences = [line]
                for sent in sentences:
                    tokens = tokenize(sent.strip())
                    if not args.min_length or len(tokens) >= args.min_length:
                        outf.write(' '.join(tokens).lower())
                        outf.write('\n')


if __name__ == '__main__':
    main()

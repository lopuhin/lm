#!/usr/bin/env python3
import argparse
import re

from nltk.tokenize import wordpunct_tokenize, sent_tokenize


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('corpus')
    arg('output')
    arg('--tokenizer', choices=['nltk', 're', 'mystem'], default='nltk')
    arg('--min-length', type=int)
    arg('--sent-tokenize', action='store_true',
        help='assume sentences do not span lines')
    args = parser.parse_args()

    if args.tokenizer == 're':
        tokenize = lambda s: re.findall('\w+', s)
    elif args.tokenizer == 'nltk':
        tokenize = wordpunct_tokenize
    elif args.tokenizer == 'mystem':
        from pymystem3 import Mystem
        mystem = Mystem()
        tokenize = lambda s: [
            x for x in (x['text'].strip() for x in mystem.analyze(s))
            if x]
    else:
        raise ValueError('Invalid tokenizer {}'.format(args.tokenizer))

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
                        outf.write(normalize(' '.join(tokens)))
                        outf.write('\n')


def normalize(text):
    return text.lower().replace('ё', 'е')


if __name__ == '__main__':
    main()

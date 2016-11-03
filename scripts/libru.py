#!/usr/bin/env python3
import argparse
import re
import os

from nltk.tokenize import sent_tokenize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('libru_root')
    parser.add_argument('output')
    args = parser.parse_args()

    encoding = 'cp1251'
    with open(args.output, 'w') as outf:
        for filename in _filenames(args.libru_root):
            print(filename)
            with open(filename, 'rb') as f:
                try:
                    text = f.read().decode(encoding)
                except UnicodeDecodeError:
                    print('skipping: wrong encoding')
                    continue
                text = re.sub('<.*?>', '', text, flags=re.S | re.M)
                if len(text) > 10000:
                    for line in sent_tokenize(text):
                        line = re.sub('\s+', ' ', line, flags=re.M).strip()
                        if line:
                            outf.write(line)
                            outf.write('\n')


def _filenames(path):
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.txt.html'):
                yield os.path.join(dirname, filename)


if __name__ == '__main__':
    main()
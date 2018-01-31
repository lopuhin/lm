#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import random

from nltk.tokenize import sent_tokenize
import tqdm


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('taiga_root')
    arg('output')
    arg('--section', help='process only this section')
    args = parser.parse_args()

    paths = []
    for section in Path(args.taiga_root).iterdir():
        print(section)
        if section.name == 'stihi_ru':
            print('skip {}'.format(section))
            continue
        if section.is_dir() and args.section in {None, section.name}:
            paths.extend((section / 'texts').glob('**/*.txt'))
    random.seed(42)
    random.shuffle(paths)

    with open(args.output, 'wt') as outf:
        for path in tqdm.tqdm(paths):
            text = path.read_text(encoding='utf8')
            for line in sent_tokenize(text):
                line = re.sub('\s+', ' ', line, flags=re.M).strip()
                if line:
                    outf.write(line)
                    outf.write('\n')


if __name__ == '__main__':
    main()

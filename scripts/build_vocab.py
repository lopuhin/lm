#!/usr/bin/env python3

import argparse
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('output')
    arg('limit', type=int)
    arg('filenames', nargs='+')
    args = parser.parse_args()

    s = '<S>'
    unk = '<UNK>'
    counts = Counter()
    for filename in args.filenames:
        print('Reading', filename)
        with open(filename) as f:
            for line in f:
                line = line.strip()
                counts.update(line.split())
                counts[s] += 1

    token_counts = counts.most_common()
    if len(token_counts) > args.limit:
        unk_count = sum(c for _, c in token_counts[args.limit:])
        total_count = sum(c for _, c in token_counts)
        print('OOV rate: {:.2%}, min count {}'.format(
            unk_count / total_count, token_counts[args.limit][1]))
        token_counts = token_counts[:args.limit - 1]
        token_counts.append((unk, unk_count))
        token_counts.sort(key=lambda x: x[1], reverse=True)

    with open(args.output, 'w') as outf:
        for w, count in token_counts:
            outf.write('{}\t{}\n'.format(w, count))


if __name__ == '__main__':
    main()

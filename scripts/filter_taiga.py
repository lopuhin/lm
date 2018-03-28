#!/usr/bin/env python3
import argparse
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('output_folder')
    arg('inputs', nargs='+')
    args = parser.parse_args()
    output_folder = Path(args.output_folder)
    assert output_folder.is_dir()

    for input_path in map(Path, args.inputs):
        output_path = output_folder / input_path.name
        print(f'{input_path} -> {output_path}')
        with input_path.open('rt') as f:
            with output_path.open('wt') as outf:
                for line in f:
                    if looks_good(line):
                        outf.write(line)


def looks_good(line: str) -> bool:
    n_russian = len(re.findall('[а-я]', line.lower()))
    return 2 * n_russian > len(line)


if __name__ == '__main__':
    main()

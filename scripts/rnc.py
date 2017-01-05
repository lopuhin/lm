#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import sys
from xml.etree import ElementTree


def iter_lines(f, skip_header=False):
    root = ElementTree.parse(f).getroot()
    ns_match = re.match('\{.*\}', root.tag)
    ns = ns_match.group(0) if ns_match else ''
    body = root.find('{}body'.format(ns))

    if any(el.get('content') == 'manual' for el in root.findall('.//meta')):
        def lines():
            for se in body.findall('.//se'):
                line = ' '.join(
                    w for w in (w.replace('`', '').strip() for w in se.itertext())
                    if w).strip()
                if line:
                    yield line

    else:
        def lines():
            for line in body.itertext():
                line = line.strip()
                if line:
                    yield line

    seen_header = not skip_header
    for l in lines():
        if seen_header:
            yield l
        seen_header = True


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('root', type=Path, help='RNC sub-folder')
    arg('output', type=Path, help='Text file')
    arg('--skip-header', action='store_true',
        help='skip first line (for main corpus)')
    args = parser.parse_args()

    with args.output.open('wt', encoding='utf8') as out_f:
        for p in Path(args.root).glob('**/*'):
            if not (p.name.endswith('.xhtml') or p.name.endswith('.xml')):
                continue
            with p.open('rb') as f:
                try:
                    for line in iter_lines(f):
                        out_f.write(line)
                        out_f.write('\n')
                except Exception:
                    print('Error reading {}'.format(p), file=sys.stderr)
                    raise


if __name__ == '__main__':
    main()

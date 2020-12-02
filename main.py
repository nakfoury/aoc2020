#!/usr/bin/env python3

import argparse
from collections import defaultdict

from util import getInput
from solutions import day1

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--day', action='store', type=int, help='specify puzzle day <1...25>')
parser.add_argument('-b', action='store_true', help='switch to part B')

args = parser.parse_args()

solutions = defaultdict(None)  # Dictionary of puzzle-day solution functions
solutions[1] = day1.day1

def main():
    inp = getInput(args.day)
    result = solutions[args.day](inp, args.b)
    print(result)
    return


if __name__ == '__main__':
    main()

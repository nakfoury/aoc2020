#!/usr/bin/env python3

import argparse
from collections import defaultdict

import util
import solutions

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--day', action='store', type=int, help='specify puzzle day <1...25>')
parser.add_argument('-b', action='store_true', help='switch to part B')

args = parser.parse_args()

s = defaultdict(None)  # Dictionary of puzzle-day solution functions
s[1] = solutions.day1
s[2] = solutions.day2


def run_solution(d, b=False):
    inp = util.getInput(d)
    result = s[d](inp, b)
    print("Day {}{}: {}".format(d, ('b' if b else 'a'), result))


def main():
    if args.day is None:
        for d in s:
            run_solution(d)
            run_solution(d, True)
    else:
        run_solution(args.day, args.b)
    return


if __name__ == '__main__':
    main()

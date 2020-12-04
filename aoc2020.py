#!/usr/bin/env python3

import argparse
from collections import defaultdict
from datetime import date

import util

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--day', action='store', type=int, help='specify puzzle day <1...25>')
parser.add_argument('-b', action='store_true', help='switch to part B')

args = parser.parse_args()


def get_solutions():  # Populate a dict of puzzle-day solution functions
    s = defaultdict(None)
    dn = date.today().strftime('%d').lstrip('0')
    for d in range(1, int(dn) + 2):
        s[d] = eval("solutions.day" + str(d))
    return s


def run_solution(f, b=False):  # Compute a particular solution
    inp = [x.strip('\n') for x in util.getInput(f)]
    print("Day {}{}: {}".format(f, 'b' if b else 'a', f(inp, b) if f else "No Solution Found"))


def main():
    s = get_solutions()
    if args.day is None:  # Compute all the available solutions
        for d in s:
            run_solution(d)
            run_solution(d, True)
    else:
        run_solution(s[args.day], args.b)
    return


if __name__ == '__main__':
    main()

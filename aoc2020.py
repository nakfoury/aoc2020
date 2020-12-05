#!/usr/bin/env python3

import argparse
from collections import defaultdict
from datetime import date

import util
import solutions

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--day', action='store', type=int, help='specify puzzle day <1...25>')
parser.add_argument('-b', action='store_true', help='switch to part B')

args = parser.parse_args()


s = defaultdict(None)
dn = date.today().strftime('%d').lstrip('0')
for d in range(1, int(dn) + 2):
    s[d] = eval("solutions.day" + str(d))


def run_solution(d, b=False):  # Compute a particular solution
    inp = util.getInput(d)
    ans = "Not yet available" if (not d or not inp) else s[d]([x.strip('\n') for x in inp], b)
    print("Day {}.{}: {}".format(d, '1' if b else '2', ans))


def main():
    if args.day is None:  # Compute all the avaialable solutions
        for d in s:
            run_solution(d)
            run_solution(d, True)
    else:
        run_solution(args.day, args.b)
    return


if __name__ == '__main__':
    main()

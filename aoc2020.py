#!/usr/bin/env python3

import argparse

import util
from solutions import solutions as s

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--day', action='store', type=int, help='specify puzzle day <1...25>')
parser.add_argument('-b', action='store_true', help='switch to part B')
parser.add_argument('-i', '--input', action='store', type=str, help='specify override input file')

args = parser.parse_args()


def run_solution(day, solution, inp, b=False):  # Compute and print a particular puzzle-day solution
    output = "Not yet available" if (not day or not inp) else solution([x.strip('\n') for x in inp], b)
    print("Day {}.{}: {}".format(day, '2' if b else '1', output))


def main():
    if args.day is None:  # Compute all the available solutions
        for d in s:
            inp = util.getInput(d)
            run_solution(d, s[d], inp)
            run_solution(d, s[d], inp, True)
    else:  # Compute the specified solution
        if args.day in s:
            inp = open(args.input, 'r').readlines() if args.input else util.getInput(args.day)
            run_solution(args.day, s[args.day], inp, args.b)
    return


if __name__ == '__main__':
    main()

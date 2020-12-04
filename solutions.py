import re


# --- Day 1: Report Repair --- #
def day1(input, b=False):
    inp = list(map(int, input))
    if not b:
        d = list(map(lambda x: 2020 - int(x), inp))
        for n in d:
            if n in inp:
                return n * (2020 - n)
    else:
        for x in inp:
            for y in inp:
                if 2020 - x - y in inp:
                    return x * y * (2020 - x - y)
    return None


# --- Day 2: Password Philosophy --- #
def day2(input, b=False):
    result = 0
    for pwd in input:
        m = re.search(r'(\d+)-(\d+) (\w): (\w+)', pwd)
        if m:
            if not b:
                if int(m.group(1)) <= m.group(4).count(m.group(3)) <= int(m.group(2)):
                    result += 1
            else:
                if (m.group(4)[int(m.group(1)) - 1] == m.group(3)) != (m.group(4)[int(m.group(2)) - 1] == m.group(3)):
                    result += 1
    return result


# --- Day 3: Toboggan Trajectory --- #
def day3(inp, b=False):
    return eval_slope(inp, 1, 1) * eval_slope(inp, 3, 1) * eval_slope(inp, 5, 1) * eval_slope(inp, 7, 1) * eval_slope(
        inp, 1, 2) if b else eval_slope(inp, 3, 1)


def eval_slope(inp, r, d):
    result = 0
    for i, row in enumerate(inp):
        if i % d > 0:
            continue
        if row[((r * int(i / d)) % len(row))] == '#':
            result += 1
    return result


# --- Day 4 --- #
def day4(inp, b=False):
    return

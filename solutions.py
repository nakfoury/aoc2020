import re
import copy


# --- Day 1: Report Repair --- #
def day1(inp, b=False):
    inp = list(map(int, inp))
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
def day2(inp, b=False):
    result = 0
    for pwd in inp:
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


# --- Day 4: Passport Processing --- #
def day4(inp, b=False):
    result = 0
    p = 0
    valid = {'byr', 'iyr', 'eyr', 'hgt', 'hcl', 'ecl', 'pid'}
    expression = r'(\w+):(#?\w+)' if b else r'(\w+):'
    inp.append('')
    g = (i for i, e in enumerate(inp) if e == '')

    for n in g:
        passport = ' '.join(inp[p:n])
        m = re.findall(expression, passport)

        if not b:
            fields = set(m)
            if fields.issuperset(valid):
                result += 1
        else:
            fields = dict(m)
            if set(fields.keys()).issuperset(valid):
                if validate_fields(fields):
                    result += 1
        p = n

    return result


def validate_fields(f):
    if int(f['byr']) < 1920 or int(f['byr']) > 2002:
        return False
    if int(f['iyr']) < 2010 or int(f['iyr']) > 2020:
        return False
    if int(f['eyr']) < 2020 or int(f['eyr']) > 2030:
        return False
    if f['hgt'][-2:] == 'cm':
        if int(f['hgt'][:-2]) < 150 or int(f['hgt'][:-2]) > 193:
            return False
    elif f['hgt'][-2:] == 'in':
        if int(f['hgt'][:-2]) < 59 or int(f['hgt'][:-2]) > 76:
            return False
    else:
        return False
    if not re.match(r'#[0-9a-f]{6}', f['hcl']):
        return False
    if f['ecl'] not in ['amb', 'blu', 'brn', 'gry', 'grn', 'hzl', 'oth', ]:
        return False
    if not re.match(r'\d{9}', f['pid']):
        return False
    return True


# --- Day 5: Binary Boarding --- #
def day5(inp, b=False):
    seats = []
    for boarding_pass in inp:
        row = binary_search(boarding_pass[:7], list(range(128)))
        col = binary_search(boarding_pass[-3:], list(range(8)))
        seats.append(row * 8 + col)
    return find_seat(sorted(seats)) if b else max(seats)


def binary_search(seq, rng):
    if seq == '':
        return rng[0]
    if seq[0] == 'F' or seq[0] == 'L':
        return binary_search(seq[1:], rng[:int(len(rng) / 2)])
    elif seq[0] == 'B' or seq[0] == 'R':
        return binary_search(seq[1:], rng[int(len(rng) / 2):])


def find_seat(seats):
    for i, seat in enumerate(seats):
        if seat + 1 != seats[i + 1]:
            return seat + 1
    return -1


# --- Day 6: Custom Customs --- #
def day6(inp, b=False):
    result, p = 0, 0
    f = set.union if not b else set.intersection
    inp.append('')
    g = (i for i, e in enumerate(inp) if e == '')
    for n in g:
        group = list(map(set, inp[p:n]))
        result += len(f(*group))
        p = n + 1
    return result


# --- Day 7: Handy Haversacks --- #
def day7(inp, b=False):
    rules = {}
    for r in inp:
        rules[re.search(r'(.*) bags contain', r).group(1)] = re.findall(r'(?:(no|\d+) (.*?) bags?[,.])', r)

    if not b:
        colors = []
        add_parents(rules, 'shiny gold', colors)
        return len(colors)
    else:
        return count_children(rules, 'shiny gold') - 1


def add_parents(rules, c, color_list):
    for r in rules:
        for item in rules[r]:
            if item[1] == c and r not in color_list:
                color_list.append(r)
                add_parents(rules, r, color_list)
    return


def count_children(rules, c):
    x = 0
    for count, color in rules[c]:
        if count == 'no':
            return 1
        else:
            x += int(count) * count_children(rules, color)
    return x + 1


# --- Day 8: Handheld Halting --- #
def day8(inp, b=False):
    if not b:
        return run_handheld(inp)[0]
    else:
        for i, l in enumerate(inp):
            test_code = copy.copy(inp)
            if l[:3] == 'acc':
                continue
            elif l[:3] == 'jmp':
                test_code[i] = test_code[i].replace('jmp', 'nop')
            elif l[:3] == 'nop':
                test_code[i] = test_code[i].replace('nop', 'jmp')
            out, err = run_handheld(test_code)
            if err == 0:
                return out


def run_handheld(code):
    accumulator, cptr, pptr = 0, 0, 0
    visited = []
    while True:
        if cptr == len(code):
            return accumulator, 0
        if cptr in visited:
            return accumulator, 1
        else:
            visited.append(cptr)
        if code[cptr][:3] == 'acc':
            accumulator += int(code[cptr][4:])
            cptr += 1
        elif code[cptr][:3] == 'jmp':
            cptr += int(code[cptr][4:])
        else:
            cptr += 1


# --- Day 9: Encoding Error --- #
def day9(inp, b=False):
    xmas_code = list(map(int, inp))
    xmas_table = read_preamble(xmas_code[:25])
    for n in xmas_code[25:]:
        if not lookup_table(n, xmas_table):
            return n if not b else weakness(n, xmas_code)
        update_table(n, xmas_table)
    return -1


def read_preamble(preamble):
    table = []
    for i in range(25):
        sums = [preamble[i]] + [preamble[i] + x for x in preamble]
        sums.remove(preamble[i] * 2)
        table.append(sums)
    return table


def lookup_table(n, table):
    for i in table:
        if n in i[1:]:
            return True
    return False


def update_table(n, table):
    table.pop(0)
    for row in table:
        row.pop(1)
        row.append(n + row[0])
    table.append([n] + [n + m for m in [table[x][0] for x in range(len(table))]])
    return table


def weakness(n, lst):
    for i in range(len(lst)):
        for j in range(1, len(lst)):
            if sum(lst[i:j]) == n:
                return max(lst[i:j]) + min(lst[i:j])


# --- Day 10: Adapter Array --- #
def day10(inp, b=False):
    chain = sorted(list(map(int, inp)))
    if not b:
        ones, threes = 1, 1
        for i in range(len(chain) - 1):
            if chain[i + 1] - chain[i] == 1:
                ones += 1
            elif chain[i + 1] - chain[i] == 3:
                threes += 1
        return ones * threes
    else:
        adapter_combos = {}
        chain.insert(0, 0)
        for joltage in chain:
            previous_adapters = sorted(list(set.intersection({joltage - 1, joltage - 2, joltage - 3}, set(chain))))
            if not previous_adapters:
                adapter_combos[joltage] = 1
            else:
                adapter_combos[joltage] = sum([adapter_combos[x] for x in previous_adapters])
    return adapter_combos[chain[-1]]


# --- Day 11: ??? --- #
def day11(inp, b=False):
    return -1


solutions = {
    1: day1,
    2: day2,
    3: day3,
    4: day4,
    5: day5,
    6: day6,
    7: day7,
    8: day8,
    9: day9,
    10: day10,
    11: day11,
}

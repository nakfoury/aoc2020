import re


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


# --- Day 8: ??? --- #
def day8(inp, b):
    return 0


solutions = {
    1: day1,
    2: day2,
    3: day3,
    4: day4,
    5: day5,
    6: day6,
    7: day7,
    8: day8,
}

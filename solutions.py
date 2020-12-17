import re
import copy
import numpy as np
from functools import reduce
from itertools import product

import util


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


# --- Day 11: Seating System --- #
def day11(inp, b=False):
    chart = []
    for i, l in enumerate(inp):
        chart.append([c for c in l])
    chart = np.array(chart)
    while True:
        flips = musical_chairs(chart, b, 5 if b else 4)
        new_chart = flip_seats(copy.copy(chart), flips)
        if np.array_equal(chart, new_chart):
            return count_occupied(new_chart)
        chart = new_chart


def musical_chairs(chart, looking, tolerance):
    flips = []
    for i in range(chart.shape[0]):  # i is the row
        for j in range(chart.shape[1]):  # j is the column
            hsh = 0
            for adj in util.adjacent_eight():
                s = 1
                while True:
                    if not util.is_in_bounds(i + adj[0] * s, j + adj[1] * s, chart.shape[0], chart.shape[1]):
                        break
                    if chart[i + adj[0] * s, j + adj[1] * s] == 'L':
                        break
                    if chart[i + adj[0] * s, j + adj[1] * s] == '#':
                        hsh += 1
                        break
                    if looking:
                        s += 1
                    else:
                        break
            if (chart[i, j] == 'L' and hsh == 0) or (chart[i, j] == '#' and hsh >= tolerance):
                flips.append((i, j))
    return flips


def flip_seats(chart, flips):
    for seat in flips:
        chart[seat] = '#' if chart[seat] == 'L' else 'L'
    return chart


def count_occupied(chart):
    occupied = 0
    for i in range(chart.shape[0]):
        for j in range(chart.shape[1]):
            if chart[i, j] == '#':
                occupied += 1
    return occupied


# --- Day 12: Rain Risk --- #
def day12(inp, b=False):
    sx, sy = 0, 0
    wx, wy = 10, 1
    heading = 'E'

    headings = {
        'E': (1, 0),
        'N': (0, 1),
        'W': (-1, 0),
        'S': (0, -1),
    }

    if not b:
        for l in inp:
            if l[0] == 'N':
                sy += int(l[1:])
            if l[0] == 'S':
                sy -= int(l[1:])
            if l[0] == 'E':
                sx += int(l[1:])
            if l[0] == 'W':
                sx -= int(l[1:])

            if l[0] == 'L':
                heading = chdir(heading, int(l[1:]))
            if l[0] == 'R':
                heading = chdir(heading, 360 - int(l[1:]))

            if l[0] == 'F':
                sx += headings[heading][0] * int(l[1:])
                sy += headings[heading][1] * int(l[1:])

    else:
        for l in inp:
            if l[0] == 'N':
                wy += int(l[1:])
            if l[0] == 'S':
                wy -= int(l[1:])
            if l[0] == 'E':
                wx += int(l[1:])
            if l[0] == 'W':
                wx -= int(l[1:])

            if l[0] == 'L':
                wx, wy = util.rotate_coord(wx, wy, int(l[1:]))
            if l[0] == 'R':
                wx, wy = util.rotate_coord(wx, wy, 360 - int(l[1:]))

            if l[0] == 'F':
                sx += wx * int(l[1:])
                sy += wy * int(l[1:])

    return abs(sx) + abs(sy)


def chdir(heading, deg):
    compass = 'ENWS'
    return compass[int((compass.index(heading) + deg / 360 * 4) % 4)]


# --- Day 13: Shuttle Search --- #
def day13(inp, b=False):
    if not b:
        delays = sorted([(int(x) - int(inp[0]) % int(x), int(x)) for x in inp[1].split(',') if x != 'x'],
                        key=lambda x: x[0])
        return delays[0][0] * delays[0][1]
    else:
        sched = {}
        result = 0
        for t, bus in enumerate(inp[1].split(',')):
            if bus != 'x':
                sched[t] = int(bus)
        prod = reduce(lambda x, y: x * y, sched.values())
        for time in sched:
            result += find_mod_factor(time, sched[time], int(prod / sched[time]))
        return result % prod


def find_mod_factor(n, mod, subprod):
    factor = 1
    congr = subprod % mod
    while True:
        if (mod - (n % mod)) % mod == (congr * factor) % mod:
            return factor * subprod
        factor += 1


# --- Day 14: Docking Data --- #
def day14(inp, b=False):
    mem = {}
    zeroes, ones = 0, 0
    if not b:
        for l in inp:
            if l[:4] == 'mask':
                zeroes, ones = convert_masks(l[-36:])
            else:
                mem[re.match(r'mem\[(\d+)\]', l).group(1)] = int(l.split(' = ')[1]) & zeroes | ones
    else:
        for l in inp:
            if l[:4] == 'mask':
                mask = l[-36:]
            else:
                addrs = mask_with_floating_values(int(re.match(r'mem\[(\d+)\]', l).group(1)), mask)
                for addr in addrs:
                    mem[addr] = int(l.split(' = ')[1])
    return sum(mem.values())


def convert_masks(s):
    zeroes = int(s.replace('X', '1'), 2)
    ones = int(s.replace('X', '0'), 2)
    return zeroes, ones


def mask_with_floating_values(addr, m):
    addrs = []
    base_addr = list(format(addr | int(m.replace('X', '0'), 2), "036b"))
    floating_indices = [i for i, c in enumerate(m) if c == 'X']
    floating_values = product(range(2), repeat=m.count('X'))

    for i in floating_indices:
        base_addr[i] = '{}'
    masked_base_str = reduce(lambda x, y: x + y, base_addr)

    for v in floating_values:
        addrs.append(masked_base_str.format(*v))
    return addrs


# --- Day 15: Rambunctious Recitation --- #
def day15(inp, b=False):
    if not b:
        numbers = list(map(int, inp[0].split(',')))
        numbers.reverse()
        while len(numbers) < 2020:
            numbers.insert(0, numbers[1:].index(numbers[0]) + 1 if numbers[0] in numbers[1:] else 0)
        return numbers[0]
    else:
        numbers = {}
        for i, n in enumerate(list(map(int, inp[0].split(',')))):
            numbers[n] = i
        n = 0
        for i in range(len(numbers), 30000000 - 1):
            if n in numbers:
                tmp = i - numbers[n]
                numbers[n] = i
                n = tmp
            else:
                numbers[n] = i
                n = 0
        return n


# --- Day 16: Ticket Translation --- #
def day16(inp, b=False):
    rules = {}
    tickets = []
    l = inp.pop(0)
    while l != '':
        name, rs = tuple(l.split(':'))
        rules[name] = [ tuple(map(int, x.split('-'))) for x in re.findall(r'(\d+-\d+)', rs) ]
        l = inp.pop(0)
    inp.pop(0)
    my_ticket = list(map(int, inp.pop(0).split(',')))
    inp.pop(0)
    inp.pop(0)
    for t in inp:
        tickets.append(list(map(int, t.split(','))))
    if not b:
        result = 0
        for ticket in tickets:
            err, _ = scan_ticket(rules, ticket)
            result += err
    else:
        tickets.append(my_ticket)
        # remove invalid tickets
        invalid_tickets = []
        for ticket in tickets:
            _, v = scan_ticket(rules, ticket)
            if not v:
                invalid_tickets.append(ticket)
        for ticket in invalid_tickets:
            tickets.remove(ticket)
        # determine possible labels
        valid_rules_by_index = {}
        for i in range(len(my_ticket)):
            valid_fields = [x for x in rules]
            for rule in rules:
                valid = True
                for ticket in tickets:
                    if not check_field(rules[rule], ticket[i]):
                        valid = False
                        break
                if not valid:
                    valid_fields.remove(rule)
            valid_rules_by_index[i] = valid_fields
        result = {}
        # assign final labels
        while True:
            for i in valid_rules_by_index:
                for r in result:
                    if r in valid_rules_by_index[i]:
                        valid_rules_by_index[i].remove(r)
                if len(valid_rules_by_index[i]) == 1:
                    result[valid_rules_by_index[i][0]] = i
            if not any(valid_rules_by_index.values()):
                break
        result = reduce(lambda x,y: x*y, [my_ticket[result[x]] if 'departure' in x else 1 for x in result])
    return result


def check_field(rule, field):
    for rng in rule:
        if field in range(int(rng[0]), int(rng[1]) + 1):
            return True
    return False


def scan_ticket(rules, ticket):
    err = 0
    valid_ticket = True
    for field in ticket:
        valid = False
        for rule in rules:
            for rng in rules[rule]:
                if field in range(int(rng[0]), int(rng[1])+1):
                    valid = True
        if not valid:
            err += field
            valid_ticket = False
    return err, valid_ticket


# --- Day 17: Conway Cubes --- #
def day17(inp, b=False):
    cubes = np.full((20,20,20), '.')
    for i, l in enumerate(inp):
        cubes[9][5+i][5:5+len(l)] = [c for c in l]

    for i in range(6):
        flips = get_flips(cubes)
        apply_flips(cubes, flips)
    return count_hashes(cubes)


def get_flips(cubes):
    neighbors = util.adjacent_six_3d()
    flips = []
    for a, layer in enumerate(cubes):
        for b, row in enumerate(layer):
            for c, col in enumerate(row):
                hashes = 0
                for n in neighbors:
                    if a + n[0] < 0 or b + n[1] < 0 or c + n[2] < 0:
                        continue
                    if a + n[0] >= 20 or b + n[1] >= 20 or c + n[2] >= 20:
                        continue
                    if cubes[a + n[0]][b + n[1]][c + n[2]] == '#':
                        hashes += 1
                if cubes[a][b][c] == '.' and hashes == 3:
                    flips.append((a, b, c))
                elif cubes[a][b][c] == '#' and hashes != 2 and hashes != 3:
                    flips.append((a, b, c))
    return flips


def apply_flips(cubes, flips):
    for flip in flips:
        cubes[flip[0]][flip[1]][flip[2]] = '.' if cubes[flip[0]][flip[1]][flip[2]] == '#' else '#'
    return cubes


def count_hashes(cubes):
    hashes = 0
    for layer in cubes:
        for row in layer:
            for col in row:
                if col == '#':
                    hashes += 1
    return hashes


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
    12: day12,
    13: day13,
    14: day14,
    15: day15,
    16: day16,
    17: day17,
}

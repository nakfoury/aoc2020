import regex as re
import copy
import numpy as np
from functools import reduce
from itertools import product, chain
from collections import defaultdict, Counter

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
    accumulator, cptr = 0, 0
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
    for l in inp:
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
        rules[name] = [tuple(map(int, x.split('-'))) for x in re.findall(r'(\d+-\d+)', rs)]
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
        result = reduce(lambda x, y: x * y, [my_ticket[result[x]] if 'departure' in x else 1 for x in result])
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
                if field in range(int(rng[0]), int(rng[1]) + 1):
                    valid = True
        if not valid:
            err += field
            valid_ticket = False
    return err, valid_ticket


# --- Day 17: Conway Cubes --- #
def day17(inp, b=False):
    if not b:
        cubes = np.full((20, 20, 20), '.')
        for i, l in enumerate(inp):
            cubes[9][5 + i][5:5 + len(l)] = [c for c in l]
    else:
        cubes = np.full((20, 20, 20, 20), '.')
        for i, l in enumerate(inp):
            cubes[9][9][5 + i][5:5 + len(l)] = [c for c in l]

    for i in range(6):
        flips = get_flips(cubes, b)
        apply_flips(cubes, flips, b)
    return count_hashes(cubes, b)


def get_flips(cubes, part2=False):
    neighbors = util.adjacent_cells_3d() if not part2 else util.adjacent_cells_4d()
    flips = []
    for a in range(len(cubes)):
        for b in range(len(cubes)):
            for c in range(len(cubes)):
                if part2:
                    for d in range(len(cubes)):
                        hashes = 0
                        for n in neighbors:
                            if a + n[0] < 0 or b + n[1] < 0 or c + n[2] < 0 or d + n[3] < 0:
                                continue
                            if a + n[0] >= 20 or b + n[1] >= 20 or c + n[2] >= 20 or d + n[3] >= 20:
                                continue
                            if cubes[a + n[0]][b + n[1]][c + n[2]][d + n[3]] == '#':
                                hashes += 1
                        if cubes[a][b][c][d] == '.' and hashes == 3:
                            flips.append((a, b, c, d))
                        elif cubes[a][b][c][d] == '#' and hashes != 2 and hashes != 3:
                            flips.append((a, b, c, d))
                else:
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


def apply_flips(cubes, flips, b=False):
    if not b:
        for flip in flips:
            cubes[flip[0]][flip[1]][flip[2]] = '.' if cubes[flip[0]][flip[1]][flip[2]] == '#' else '#'
    else:
        for flip in flips:
            cubes[flip[0]][flip[1]][flip[2]][flip[3]] = '.' if cubes[flip[0]][flip[1]][flip[2]][flip[3]] == '#' else '#'
    return cubes


def count_hashes(cubes, b=False):
    hashes = 0
    for layer in cubes:
        for row in layer:
            for col in row:
                if b:
                    for h in col:
                        if h == '#':
                            hashes += 1
                elif col == '#':
                    hashes += 1
    return hashes


# --- Day 18: Operation Order --- #
def day18(inp, b=False):
    problems = [bytearray(l, 'UTF-8') for l in inp]
    result = 0
    for p in problems:
        result += int(evaluate(p, b))
    return result


def evaluate(exp, b):
    m = re.search(r'(\([^\(]*?\))', exp.decode())
    while m:
        exp[m.start():m.end()] = evaluate(bytearray(m.group(0)[1:-1], 'UTF-8'), b)
        m = re.search(r'(\([^\(]*?\))', exp.decode())

    if not b:
        m = re.search(r'(\d+ [\*\+] \d+)', exp.decode())
        while m:
            exp[m.start():m.end()] = bytearray(str(eval(m.group(0))), 'UTF-8')
            m = re.search(r'(\d+ [\*\+] \d+)', exp.decode())

    else:
        m = re.search(r'(\d+ [\+] \d+)', exp.decode())
        while m:
            exp[m.start():m.end()] = bytearray(str(eval(m.group(0))), 'UTF-8')
            m = re.search(r'(\d+ [\+] \d+)', exp.decode())
        m = re.search(r'(\d+ [\*] \d+)', exp.decode())
        while m:
            exp[m.start():m.end()] = bytearray(str(eval(m.group(0))), 'UTF-8')
            m = re.search(r'(\d+ [\*] \d+)', exp.decode())

    return exp


# --- Day 19: Monster Messages --- #
def day19(inp, b=False):
    rules = {}
    for l in inp[:inp.index('')]:
        rules[l.split(':')[0]] = l.split(': ')[1]
    m = re.search(r'\d+', rules['0'])
    while m:
        if b and m.group(0) == '8':
            rules['0'] = rules['0'][:m.start()] + '(42+)' + rules['0'][m.end():]
        elif b and m.group(0) == '11':
            rules['0'] = rules['0'][:m.start()] + '(?P<recur>42 (?&recur)? 31)' + rules['0'][m.end():]
        else:
            rules['0'] = rules['0'][:m.start()] + '(' + rules[m.group(0)].strip('"') + ')' + rules['0'][m.end():]
        m = re.search(r'\d+', rules['0'])
    exp = re.compile(rules['0'].replace(' ', '').encode('unicode-escape').decode())
    return sum(1 for x in map(exp.fullmatch, inp[inp.index(''):]) if x is not None)


# --- Day 20: Jurassic Jigsaw --- #
def day20(inp, b=False):
    result = 0
    tiles = {}
    while inp:
        t = re.search(r'\d+', inp.pop(0)).group(0)
        buf = []
        while True:
            l = inp.pop(0)
            if l == '':
                break
            buf.append([c for c in l])
        tiles[t] = np.array(buf)
    if not b:
        result = 1
        matches = count_matches_per_tile(tiles)
        for t in matches:
            if matches[t] == 4:
                result *= int(t)
    return result


def count_matches_per_tile(tiles):
    matches = defaultdict(int)
    for t1 in tiles:
        e1 = get_edges(tiles[t1])
        for t2 in tiles:
            if t1 == t2:
                continue
            e2 = get_edges(tiles[t2])
            for edge1 in e1:
                for edge2 in e2:
                    if (edge1 == edge2).all():
                        matches[t1] += 1
    return matches


def get_edges(tile):
    result = [tile[:, 0], tile[0, :], tile[:, 9], tile[9, :]]
    result += [np.flip(x) for x in result]
    return result


# --- Day 21: Allergen Assessment --- #
def day21(inp, b=False):
    allergens = {}
    ingredients = Counter()
    result = 0
    for l in inp:
        for a in re.search(r'\(contains (.+)\)', l).group(1).split(', '):
            if a not in allergens:
                allergens[a] = l[:l.index('(')].split()
            else:
                removals = []
                for ing in allergens[a]:
                    if ing not in l[:l.index('(')].split():
                        removals.append(ing)
                for r in removals:
                    allergens[a].remove(r)
        ingredients.update(l[:l.index('(')].split())

    if not b:
        all_allergens = list(set(chain.from_iterable(allergens.values())))
        for ingredient in ingredients:
            if ingredient not in all_allergens:
                result += ingredients[ingredient]
    else:
        solve_puzzle(allergens)
        result = reduce(lambda x, y: x + ',' + y, [allergens[x][0] for x in sorted(allergens.keys())])
    return result


def solve_puzzle(d):
    while any([len(d[x]) > 1 for x in d]):
        for a in d:
            all_elts = list(chain.from_iterable(d.values()))
            for i in d[a]:
                if all_elts.count(i) == 1:
                    d[a] = [i]
                    break


# --- Day 22: Crab Combat --- #
def day22(inp, b=False):
    p1deck = list(map(int, inp[inp.index('Player 1:') + 1:inp.index('')]))
    p2deck = list(map(int, inp[inp.index('Player 2:') + 1:]))
    previous_states = []
    while True:
        if p1deck == []:
            return score(p2deck)
        elif p2deck == []:
            return score(p1deck)
        if not b:
            combat_round(p1deck, p2deck)
        else:
            return score(p1deck) if recursive_combat_subgame(p1deck, p2deck) == 'p1' else score(p2deck)


def score(deck):
    deck.reverse()
    return sum([i * x for i, x in enumerate(deck, start=1)])


def combat_round(p1, p2):
    p1card = p1.pop(0)
    p2card = p2.pop(0)
    if int(p1card) > int(p2card):
        p1.append(p1card)
        p1.append(p2card)
    else:
        p2.append(p2card)
        p2.append(p1card)


def recursive_combat_subgame(p1deck, p2deck):
    previous_states = []
    while True:
        if p1deck == []:
            return 'p2'
        elif p2deck == []:
            return 'p1'
        if (str(p1deck), str(p2deck)) in previous_states:
            return 'p1'
        previous_states.append((str(p1deck), str(p2deck)))
        recursive_combat_round(p1deck, p2deck)


def recursive_combat_round(p1deck, p2deck):
    if p1deck[0] > len(p1deck[1:]) or p2deck[0] > len(p2deck[1:]):
        combat_round(p1deck, p2deck)
    else:
        p1card = p1deck.pop(0)
        p2card = p2deck.pop(0)
        winner = recursive_combat_subgame(copy.copy(p1deck[:p1card]), copy.copy(p2deck[:p2card]))
        if winner == 'p1':
            p1deck.append(p1card)
            p1deck.append(p2card)
        if winner == 'p2':
            p2deck.append(p2card)
            p2deck.append(p1card)


# --- Day 23: Crab Cups --- #
def day23(inp, b=False):
    if not b:
        cups = [int(c) for c in inp[0]]
    else:
        cups = [int(c) for c in range(1, 1000001)]
        cups[:len(inp[0])] = [int(c) for c in inp[0]]
    cups.reverse()
    cups = DLL(cups)
    rounds = 100 if not b else 10000000
    for _ in range(rounds):
        cups.pick_up_cups()
        cups.choose_destination()
        cups.place_cups()
        cups.rotate_clockwise()
    return cups.print_cups(b)


class Cup:
    def __init__(self, id):
        self.id = id
        self.next = None
        self.prev = None


class DLL:
    def __init__(self, inp) -> object:
        self.current_cup = None
        self.held_cups = []
        self.destination_id = None
        self.lowest_cup = min(inp)
        self.highest_cup = max(inp)
        self.tail = None
        self.cup_table = {}
        for cup in inp:
            self.insert_at_head(cup)

    def print_cups(self, b=False):
        if b:
            first = self.get_cup_by_id(1)
            return first.next.id * first.next.next.id
        else:
            result = []
            cup_ptr = self.get_cup_by_id(1).next
            for _ in range(8):
                result.append(str(cup_ptr.id))
                cup_ptr = cup_ptr.next
            return (''.join(result))

    def rotate_clockwise(self):
        self.current_cup = self.current_cup.next

    def rotate_counterclockwise(self):
        self.current_cup = self.current_cup.prev

    def pick_up_cups(self):
        for _ in range(3):
            self.held_cups.append(self.current_cup.next)
            self.current_cup.next = self.current_cup.next.next
        self.current_cup.next.prev = self.current_cup

    def choose_destination(self):
        for d in range(1, 5):
            dest = self.current_cup.id - d

            if dest < self.lowest_cup:
                dest = dest + self.highest_cup

            if dest not in [c.id for c in self.held_cups]:
                self.destination_id = dest
                break

    def get_cup_by_id(self, id):
        return self.cup_table[id]

    def place_cups(self):
        cup_ptr = self.get_cup_by_id(self.destination_id)

        cup_ptr.next.prev = self.held_cups[2]
        self.held_cups[2].next = cup_ptr.next

        cup_ptr.next = self.held_cups[0]
        self.held_cups[0].prev = cup_ptr

        self.held_cups = []

    def insert_at_head(self, id):
        if self.current_cup == None:  # add 1st elt to new list
            self.current_cup = Cup(id)
            self.current_cup.next = self.current_cup
            self.current_cup.prev = self.current_cup
            self.tail = self.current_cup
        else:
            new_cup = Cup(id)

            new_cup.next = self.current_cup
            self.current_cup.prev = new_cup

            new_cup.prev = self.tail
            self.tail.next = new_cup

            self.current_cup = new_cup
        self.cup_table[id] = self.current_cup


# --- Day 24: Lobby Layout --- #
def day24(inp, b=False):
    black_tiles = []
    for l in inp:
        (x, y) = flip_tile(l)
        if (x, y) in black_tiles:
            black_tiles.remove((x, y))
        else:
            black_tiles.append((x, y))
    if b:
        for _ in range(100):
            flip_tiles(black_tiles)
    return len(black_tiles)


def flip_tile(inst):
    x, y = 0, 0
    while inst != '':
        if inst[0] == 'e':
            x += 1
            inst = inst[1:]
        elif inst[0] == 'w':
            x -= 1
            inst = inst[1:]
        elif inst[:2] == 'ne':
            if y % 2 == 1:
                x += 1
            y -= 1
            inst = inst[2:]
        elif inst[:2] == 'nw':
            if y % 2 == 0:
                x -= 1
            y -= 1
            inst = inst[2:]
        elif inst[:2] == 'se':
            if y % 2 == 1:
                x += 1
            y += 1
            inst = inst[2:]
        elif inst[:2] == 'sw':
            if y % 2 == 0:
                x -= 1
            y += 1
            inst = inst[2:]
    return x, y


def flip_tiles(black_tiles):
    new_flips = []
    white_tiles = []

    for x, y in black_tiles:
        neighbors = 0
        for n in hex_neighbors(x, y):
            if n in black_tiles:
                neighbors += 1
            elif n not in white_tiles:
                white_tiles.append(n)
        if neighbors == 0 or neighbors > 2:
            new_flips.append((x, y))

    for x, y in white_tiles:
        neighbors = 0
        for n in hex_neighbors(x, y):
            if n in black_tiles:
                neighbors += 1
        if neighbors == 2:
            new_flips.append((x, y))

    for flip in new_flips:
        if flip in black_tiles:
            black_tiles.remove(flip)
        else:
            black_tiles.append(flip)
    return


def hex_neighbors(x, y):
    if y % 2 == 0:
        return [(x - 1, y), (x + 1, y), (x - 1, y - 1), (x - 1, y + 1), (x, y + 1), (x, y - 1)]
    else:
        return [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1), (x + 1, y + 1), (x + 1, y - 1)]


# --- Day 25: Combo Breaker --- #
def day25(inp, b=False):
    door_pub = int(inp[0])
    card_pub = int(inp[1])
    door_loop = transform_subject_number(7, target=door_pub)
    card_loop = transform_subject_number(7, target=card_pub)
    return transform_subject_number(door_pub, loop_size=card_loop)


def transform_subject_number(num, target=None, loop_size=None):
    result = 1
    if loop_size:
        for i in range(loop_size):
            result = (result * num) % 20201227
    else:
        cycle = 0
        while True:
            if result == target:
                result = cycle
                break
            result = (result * num) % 20201227
            cycle += 1
    return result


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
    18: day18,
    19: day19,
    20: day20,
    21: day21,
    22: day22,
    23: day23,
    24: day24,
    25: day25,
}

import solutions


def test_day1a():
    assert solutions.day1(['1721', '979', '366', '299', '675', '1456']) == 514579


def test_day1b():
    assert solutions.day1(['1721', '979', '366', '299', '675', '1456'], True) == 241861950


def test_day2a():
    assert solutions.day2(['1-3 a: abcde', '1-3 b: cdefg', '2-9 c: ccccccccc']) == 2


def test_day2b():
    assert solutions.day2(['1-3 a: abcde', '1-3 b: cdefg', '2-9 c: ccccccccc'], True) == 1


def test_day3a():
    assert solutions.day3(['..##.........##.........##.........##.........##.........##.......',
                           '#...#...#..#...#...#..#...#...#..#...#...#..#...#...#..#...#...#..',
                           '.#....#..#..#....#..#..#....#..#..#....#..#..#....#..#..#....#..#.',
                           '..#.#...#.#..#.#...#.#..#.#...#.#..#.#...#.#..#.#...#.#..#.#...#.#',
                           '.#...##..#..#...##..#..#...##..#..#...##..#..#...##..#..#...##..#.',
                           '..#.##.......#.##.......#.##.......#.##.......#.##.......#.##.....',
                           '.#.#.#....#.#.#.#....#.#.#.#....#.#.#.#....#.#.#.#....#.#.#.#....#',
                           '.#........#.#........#.#........#.#........#.#........#.#........#',
                           '#.##...#...#.##...#...#.##...#...#.##...#...#.##...#...#.##...#...',
                           '#...##....##...##....##...##....##...##....##...##....##...##....#',
                           '.#..#...#.#.#..#...#.#.#..#...#.#.#..#...#.#.#..#...#.#.#..#...#.#', ]) == 7
    assert solutions.day3(['..##.......',
                           '#...#...#..',
                           '.#....#..#.',
                           '..#.#...#.#',
                           '.#...##..#.',
                           '..#.##.....',
                           '.#.#.#....#',
                           '.#........#',
                           '#.##...#...',
                           '#...##....#',
                           '.#..#...#.#', ]) == 7


def test_day3b():
    assert solutions.day3(['..##.........##.........##.........##.........##.........##.......',
                           '#...#...#..#...#...#..#...#...#..#...#...#..#...#...#..#...#...#..',
                           '.#....#..#..#....#..#..#....#..#..#....#..#..#....#..#..#....#..#.',
                           '..#.#...#.#..#.#...#.#..#.#...#.#..#.#...#.#..#.#...#.#..#.#...#.#',
                           '.#...##..#..#...##..#..#...##..#..#...##..#..#...##..#..#...##..#.',
                           '..#.##.......#.##.......#.##.......#.##.......#.##.......#.##.....',
                           '.#.#.#....#.#.#.#....#.#.#.#....#.#.#.#....#.#.#.#....#.#.#.#....#',
                           '.#........#.#........#.#........#.#........#.#........#.#........#',
                           '#.##...#...#.##...#...#.##...#...#.##...#...#.##...#...#.##...#...',
                           '#...##....##...##....##...##....##...##....##...##....##...##....#',
                           '.#..#...#.#.#..#...#.#.#..#...#.#.#..#...#.#.#..#...#.#.#..#...#.#', ], b=True) == 336
    assert solutions.day3(['..##.......',
                           '#...#...#..',
                           '.#....#..#.',
                           '..#.#...#.#',
                           '.#...##..#.',
                           '..#.##.....',
                           '.#.#.#....#',
                           '.#........#',
                           '#.##...#...',
                           '#...##....#',
                           '.#..#...#.#', ], b=True) == 336


def test_day4a():
    assert solutions.day4(['ecl:gry pid:860033327 eyr:2020 hcl:#fffffd',
                           'byr:1937 iyr:2017 cid:147 hgt:183cm',
                           '',
                           'iyr:2013 ecl:amb cid:350 eyr:2023 pid:028048884',
                           'hcl:#cfa07d byr:1929',
                           '',
                           'hcl:#ae17e1 iyr:2013',
                           'eyr:2024',
                           'ecl:brn pid:760753108 byr:1931',
                           'hgt:179cm',
                           '',
                           'hcl:#cfa07d eyr:2025 pid:166559648',
                           'iyr:2011 ecl:brn hgt:59in', '', ], b=False) == 2


def test_day4b():
    assert solutions.day4(['eyr:1972 cid:100',
                           'hcl:#18171d ecl:amb hgt:170 pid:186cm iyr:2018 byr:1926',
                           ''
                           'iyr:2019',
                           'hcl:#602927 eyr:1967 hgt:170cm',
                           'ecl:grn pid:012533040 byr:1946',
                           ''
                           'hcl:dab227 iyr:2012',
                           'ecl:brn hgt:182cm pid:021572410 eyr:2020 byr:1992 cid:277',
                           ''
                           'hgt:59cm ecl:zzz',
                           'eyr:2038 hcl:74454a iyr:2023',
                           'pid:3556412378 byr:2007', '', ], b=True) == 0

    assert solutions.day4(['pid:087499704 hgt:74in ecl:grn iyr:2012 eyr:2030 byr:1980',
                           'hcl:#623a2f',
                           '',
                           'eyr:2029 ecl:blu cid:129 byr:1989',
                           'iyr:2014 pid:896056539 hcl:#a97842 hgt:165cm',
                           '',
                           'hcl:#888785',
                           'hgt:164cm byr:2001 iyr:2015 cid:88',
                           'pid:545766238 ecl:hzl',
                           'eyr:2022',
                           '',
                           'iyr:2010 hgt:158cm hcl:#b6652a ecl:blu byr:1944 eyr:2021 pid:093154719', '', ], b=True) == 4


def test_day5a():
    assert solutions.day5(['BFFFBBFRRR']) == 567


def test_day5b():
    assert solutions.day5(['BFFFBBFRLL', 'BFFFBBFRRL', 'BFFFBBFRRR'], True) == 565


def test_day6a():
    assert solutions.day6(['abc',
                           '',
                           'a',
                           'b',
                           'c',
                           '',
                           'ab',
                           'ac',
                           '',
                           'a',
                           'a',
                           'a',
                           'a',
                           '',
                           'b', ]) == 11


def test_day6b():
    assert solutions.day6(['abc',
                           '',
                           'a',
                           'b',
                           'c',
                           '',
                           'ab',
                           'ac',
                           '',
                           'a',
                           'a',
                           'a',
                           'a',
                           '',
                           'b', ], b=True) == 6


def test_day7a():
    assert solutions.day7(['light red bags contain 1 bright white bag, 2 muted yellow bags.',
                           'dark orange bags contain 3 bright white bags, 4 muted yellow bags.',
                           'bright white bags contain 1 shiny gold bag.',
                           'muted yellow bags contain 2 shiny gold bags, 9 faded blue bags.',
                           'shiny gold bags contain 1 dark olive bag, 2 vibrant plum bags.',
                           'dark olive bags contain 3 faded blue bags, 4 dotted black bags.',
                           'vibrant plum bags contain 5 faded blue bags, 6 dotted black bags.',
                           'faded blue bags contain no other bags.',
                           'dotted black bags contain no other bags.', ]) == 4


def test_day7b():
    assert solutions.day7(['shiny gold bags contain 2 dark red bags.',
                           'dark red bags contain 2 dark orange bags.',
                           'dark orange bags contain 2 dark yellow bags.',
                           'dark yellow bags contain 2 dark green bags.',
                           'dark green bags contain 2 dark blue bags.',
                           'dark blue bags contain 2 dark violet bags.',
                           'dark violet bags contain no other bags.'], b=True) == 126
    assert solutions.day7(['light red bags contain 1 bright white bag, 2 muted yellow bags.',
                           'dark orange bags contain 3 bright white bags, 4 muted yellow bags.',
                           'bright white bags contain 1 shiny gold bag.',
                           'muted yellow bags contain 2 shiny gold bags, 9 faded blue bags.',
                           'shiny gold bags contain 1 dark olive bag, 2 vibrant plum bags.',
                           'dark olive bags contain 3 faded blue bags, 4 dotted black bags.',
                           'vibrant plum bags contain 5 faded blue bags, 6 dotted black bags.',
                           'faded blue bags contain no other bags.',
                           'dotted black bags contain no other bags.', ], b=True) == 32


def test_day8a():
    assert solutions.day8(['nop +0',
                           'acc +1',
                           'jmp +4',
                           'acc +3',
                           'jmp -3',
                           'acc -99',
                           'acc +1',
                           'jmp -4',
                           'acc +6', ]) == 5


def test_day8b():
    assert solutions.day8(['nop +0',
                           'acc +1',
                           'jmp +4',
                           'acc +3',
                           'jmp -3',
                           'acc -99',
                           'acc +1',
                           'jmp -4',
                           'acc +6', ], b=True) == 8


# Day 9 solution requires magic number 25 whereas TCs require 5
#
# def test_day9a():
#     assert solutions.day9(['35',
#                            '20',
#                            '15',
#                            '25',
#                            '47',
#                            '40',
#                            '62',
#                            '55',
#                            '65',
#                            '95',
#                            '102',
#                            '117',
#                            '150',
#                            '182',
#                            '127',
#                            '219',
#                            '299',
#                            '277',
#                            '309',
#                            '576', ]) == 127
#
#
# def test_day9b():
#     assert solutions.day9(['35',
#                            '20',
#                            '15',
#                            '25',
#                            '47',
#                            '40',
#                            '62',
#                            '55',
#                            '65',
#                            '95',
#                            '102',
#                            '117',
#                            '150',
#                            '182',
#                            '127',
#                            '219',
#                            '299',
#                            '277',
#                            '309',
#                            '576', ], b=True) == 62


def test_day10a():
    assert solutions.day10(['16',
                            '10',
                            '15',
                            '5',
                            '1',
                            '11',
                            '7',
                            '19',
                            '6',
                            '12',
                            '4']) == 35
    assert solutions.day10(['28',
                            '33',
                            '18',
                            '42',
                            '31',
                            '14',
                            '46',
                            '20',
                            '48',
                            '47',
                            '24',
                            '23',
                            '49',
                            '45',
                            '19',
                            '38',
                            '39',
                            '11',
                            '1',
                            '32',
                            '25',
                            '35',
                            '8',
                            '17',
                            '7',
                            '9',
                            '4',
                            '2',
                            '34',
                            '10',
                            '3', ]) == 220


def test_day10b():
    assert solutions.day10(['16',
                            '10',
                            '15',
                            '5',
                            '1',
                            '11',
                            '7',
                            '19',
                            '6',
                            '12',
                            '4'], b=True) == 8
    assert solutions.day10(['28',
                            '33',
                            '18',
                            '42',
                            '31',
                            '14',
                            '46',
                            '20',
                            '48',
                            '47',
                            '24',
                            '23',
                            '49',
                            '45',
                            '19',
                            '38',
                            '39',
                            '11',
                            '1',
                            '32',
                            '25',
                            '35',
                            '8',
                            '17',
                            '7',
                            '9',
                            '4',
                            '2',
                            '34',
                            '10',
                            '3', ], b=True) == 19208


def test_day11a():
    assert solutions.day11(['L.LL.LL.LL',
                            'LLLLLLL.LL',
                            'L.L.L..L..',
                            'LLLL.LL.LL',
                            'L.LL.LL.LL',
                            'L.LLLLL.LL',
                            '..L.L.....',
                            'LLLLLLLLLL',
                            'L.LLLLLL.L',
                            'L.LLLLL.LL', ]) == 37


def test_day11b():
    assert solutions.day11(['L.LL.LL.LL',
                            'LLLLLLL.LL',
                            'L.L.L..L..',
                            'LLLL.LL.LL',
                            'L.LL.LL.LL',
                            'L.LLLLL.LL',
                            '..L.L.....',
                            'LLLLLLLLLL',
                            'L.LLLLLL.L',
                            'L.LLLLL.LL', ], b=True) == 26


def test_day12a():
    assert solutions.chdir('E', 90) == 'N'
    assert solutions.day12(['F10',
                            'N3',
                            'F7',
                            'R90',
                            'F11', ]) == 25


def test_day12b():
    assert solutions.day12(['F10',
                            'N3',
                            'F7',
                            'R90',
                            'F11', ], b=True) == 286


def test_day13a():
    assert solutions.day13(['939',
                            '7,13,x,x,59,x,31,19', ]) == 295


def test_day13b():
    assert solutions.day13(['939', '7,13,x,x,59,x,31,19', ], b=True) == 1068781


def test_day14a():
    assert solutions.day14(['mask = XXXXXXXXXXXXXXXXXXXXXXXXXXXXX1XXXX0X',
                            'mem[8] = 11',
                            'mem[7] = 101',
                            'mem[8] = 0', ]) == 165


def test_day14b():
    assert solutions.day14(['mask = 000000000000000000000000000000X1001X',
                            'mem[42] = 100',
                            'mask = 00000000000000000000000000000000X0XX',
                            'mem[26] = 1', ], b=True) == 208


# def test_day15a():
#     assert solutions.day15(['0,3,6']) == 436
#
#
# def test_day15b():
#     assert solutions.day15(['0,3,6'], b=True) == 175594


def test_day16a():
    assert solutions.day16(['class: 1-3 or 5-7',
                            'row: 6-11 or 33-44',
                            'seat: 13-40 or 45-50',
                            '',
                            'your ticket:',
                            '7,1,14',
                            '',
                            'nearby tickets:',
                            '7,3,47',
                            '40,4,50',
                            '55,2,20',
                            '38,6,12', ]) == 71


# def test_day16b():
#     assert solutions.day16(['class: 1-3 or 5-7',
#                             'row: 6-11 or 33-44',
#                             'seat: 13-40 or 45-50',
#                             '',
#                             'your ticket:',
#                             '7,1,14',
#                             '',
#                             'nearby tickets:',
#                             '7,3,47',
#                             '40,4,50',
#                             '55,2,20',
#                             '38,6,12',], b=True) == -1


def test_day17a():
    assert solutions.day17(['.#.',
                            '..#',
                            '###', ]) == 112


# def test_day17b():
#     assert solutions.day17(['.#.',
#                             '..#',
#                             '###', ], b=True) == 848


def test_day18a():
    assert solutions.day18(['1 + 2 * 3 + 4 * 5 + 6']) == 71
    assert solutions.day18(['2 * 3 + (4 * 5)']) == 26
    assert solutions.day18(['5 + (8 * 3 + 9 + 3 * 4 * 3)']) == 437


def test_day18b():
    assert solutions.day18(['1 + (2 * 3) + (4 * (5 + 6))'], b=True) == 51
    assert solutions.day18(['2 * 3 + (4 * 5)'], b=True) == 46
    assert solutions.day18(['5 + (8 * 3 + 9 + 3 * 4 * 3)'], b=True) == 1445


def test_day19a():
    assert solutions.day19(['0: 4 1 5',
                            '1: 2 3 | 3 2',
                            '2: 4 4 | 5 5',
                            '3: 4 5 | 5 4',
                            '4: "a"',
                            '5: "b"',
                            '',
                            'ababbb',
                            'bababa',
                            'abbbab',
                            'aaabbb',
                            'aaaabbb', ]) == 2


# def test_day19b():
#     assert solutions.day19(['0: 4 1 5',
#                             '1: 2 3 | 3 2',
#                             '2: 4 4 | 5 5',
#                             '3: 4 5 | 5 4',
#                             '4: "a"',
#                             '5: "b"',
#                             '',
#                             'ababbb',
#                             'bababa',
#                             'abbbab',
#                             'aaabbb',
#                             'aaaabbb',], b) == -1


def test_day20a():
    assert solutions.day20([]) == 1


def test_day20b():
    assert solutions.day20([], b=True) == 0


def test_day21a():
    assert solutions.day21(['mxmxvkd kfcds sqjhc nhms (contains dairy, fish)',
                            'trh fvjkl sbzzf mxmxvkd (contains dairy)',
                            'sqjhc fvjkl (contains soy)',
                            'sqjhc mxmxvkd sbzzf (contains fish)', ]) == 5


def test_day21b():
    assert solutions.day21(['mxmxvkd kfcds sqjhc nhms (contains dairy, fish)',
                            'trh fvjkl sbzzf mxmxvkd (contains dairy)',
                            'sqjhc fvjkl (contains soy)',
                            'sqjhc mxmxvkd sbzzf (contains fish)', ], b=True) == 'mxmxvkd,sqjhc,fvjkl'


def test_day22a():
    assert solutions.day22(['Player 1:',
                            '9',
                            '2',
                            '6',
                            '3',
                            '1',
                            '',
                            'Player 2:',
                            '5',
                            '8',
                            '4',
                            '7',
                            '10', ]) == 306


def test_day22b():
    assert solutions.day22(['Player 1:',
                            '9',
                            '2',
                            '6',
                            '3',
                            '1',
                            '',
                            'Player 2:',
                            '5',
                            '8',
                            '4',
                            '7',
                            '10', ], b=True) == 291


def test_day23a():
    assert solutions.day23(['389125467']) == '67384529'


# def test_day23b():
#     assert solutions.day23(['389125467'], b=True) == 149245887792


def test_day24a():
    assert solutions.day24(['sesenwnenenewseeswwswswwnenewsewsw',
                            'neeenesenwnwwswnenewnwwsewnenwseswesw',
                            'seswneswswsenwwnwse',
                            'nwnwneseeswswnenewneswwnewseswneseene',
                            'swweswneswnenwsewnwneneseenw',
                            'eesenwseswswnenwswnwnwsewwnwsene',
                            'sewnenenenesenwsewnenwwwse',
                            'wenwwweseeeweswwwnwwe',
                            'wsweesenenewnwwnwsenewsenwwsesesenwne',
                            'neeswseenwwswnwswswnw',
                            'nenwswwsewswnenenewsenwsenwnesesenew',
                            'enewnwewneswsewnwswenweswnenwsenwsw',
                            'sweneswneswneneenwnewenewwneswswnese',
                            'swwesenesewenwneswnwwneseswwne',
                            'enesenwswwswneneswsenwnewswseenwsese',
                            'wnwnesenesenenwwnenwsewesewsesesew',
                            'nenewswnwewswnenesenwnesewesw',
                            'eneswnwswnwsenenwnwnwwseeswneewsenese',
                            'neswnwewnwnwseenwseesewsenwsweewe',
                            'wseweeenwnesenwwwswnew', ]) == 10


# def test_day24b():
#     assert solutions.day24(['sesenwnenenewseeswwswswwnenewsewsw',
#                             'neeenesenwnwwswnenewnwwsewnenwseswesw',
#                             'seswneswswsenwwnwse',
#                             'nwnwneseeswswnenewneswwnewseswneseene',
#                             'swweswneswnenwsewnwneneseenw',
#                             'eesenwseswswnenwswnwnwsewwnwsene',
#                             'sewnenenenesenwsewnenwwwse',
#                             'wenwwweseeeweswwwnwwe',
#                             'wsweesenenewnwwnwsenewsenwwsesesenwne',
#                             'neeswseenwwswnwswswnw',
#                             'nenwswwsewswnenenewsenwsenwnesesenew',
#                             'enewnwewneswsewnwswenweswnenwsenwsw',
#                             'sweneswneswneneenwnewenewwneswswnese',
#                             'swwesenesewenwneswnwwneseswwne',
#                             'enesenwswwswneneswsenwnewswseenwsese',
#                             'wnwnesenesenenwwnenwsewesewsesesew',
#                             'nenewswnwewswnenesenwnesewesw',
#                             'eneswnwswnwsenenwnwnwwseeswneewsenese',
#                             'neswnwewnwnwseenwseesewsenwsweewe',
#                             'wseweeenwnesenwwwswnew',], b=True) == 2208


def test_day25():
    assert solutions.transform_subject_number(7, 17807724) == 11
    assert solutions.transform_subject_number(7, 5764801) == 8
    assert solutions.transform_subject_number(17807724, loop_size=8) == 14897079
    assert solutions.transform_subject_number(5764801, loop_size=11) == 14897079

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
                    return x * y * (2020 - x -y)
    return None

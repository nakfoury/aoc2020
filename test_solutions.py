from solutions import day1, day2


def test_day1():
    assert day1.day1(['1721','979','366','299','675','1456']) == 514579
    assert day1.day1(['1721','979','366','299','675','1456'], True) == 241861950


def test_day2():
    assert day2.day2(['1-3 a: abcde','1-3 b: cdefg','2-9 c: ccccccccc']) == 2
    assert day2.day2(['1-3 a: abcde', '1-3 b: cdefg', '2-9 c: ccccccccc'], True) == 1

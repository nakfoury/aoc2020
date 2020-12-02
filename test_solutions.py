import solutions


def test_day1a():
    assert solutions.day1(['1721','979','366','299','675','1456']) == 514579


def test_day1b():
    assert solutions.day1(['1721','979','366','299','675','1456'], True) == 241861950


def test_day2a():
    assert solutions.day2(['1-3 a: abcde','1-3 b: cdefg','2-9 c: ccccccccc']) == 2


def test_day2b():
    assert solutions.day2(['1-3 a: abcde', '1-3 b: cdefg', '2-9 c: ccccccccc'], True) == 1

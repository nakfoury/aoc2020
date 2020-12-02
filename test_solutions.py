from solutions import day1, day2


def test_day1():
    assert day1.day1(['1721','979','366','299','675','1456']) == 514579
    assert day1.day1(['1721','979','366','299','675','1456'], True) == 241861950

from os.path import isfile

import httplib2


def getInput(day):
    if not isfile('input/input{}.txt'.format(day)):
        cookie = open('cookie.txt', 'r').read()
        h = httplib2.Http(".cache")
        (resp, content) = h.request("https://adventofcode.com/2020/day/{}/input".format(day), "GET",
                                    headers={'content-type': 'text/plain', 'Cookie': str(cookie), })
        if content.decode('utf-8') == "Please don't repeatedly request this endpoint before it unlocks! " \
                                      "The calendar countdown is synchronized with the server time; the link " \
                                      "will be enabled on the calendar the instant this puzzle becomes available.\n":
            return None
        open('input/input{}.txt'.format(day), 'w').write(content.decode('utf-8'))
    return open('input/input{}.txt'.format(day), 'r').readlines()


def adjacent_eight():
    result = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            result.append((i, j))
    result.remove((0, 0))
    return list(set(result))


def is_in_bounds(x, y, rows, cols):
    if x < 0 or x >= rows or y < 0 or y >= cols:
        return False
    return True

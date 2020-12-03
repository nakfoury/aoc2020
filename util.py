from os.path import isfile

import httplib2


def getInput(day):
    if not isfile('input/input{}.txt'.format(day)):
        cookie = open('cookie.txt', 'r').read()
        h = httplib2.Http(".cache")
        (resp, content) = h.request("https://adventofcode.com/2020/day/{}/input".format(day), "GET",
                                    headers={'content-type': 'text/plain', 'Cookie': str(cookie), })
        open('input/input{}.txt'.format(day), 'w').write(content.decode('utf-8'))
    return open('input/input{}.txt'.format(day), 'r').readlines()

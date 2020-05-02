import sys, os

replaces = dict()

def walkdir1(dirname):
    for cur, _dirs, files in os.walk(dirname):
        pref = ''
        print(cur)
        # head, tail = os.path.split(cur)
        # while head:
        #     pref += '---'
        #     head, _tail = os.path.split(head)
        # print(pref+tail)
        for f in files:
            if f == 'CMakeLists.txt':
                continue
            f = f.replace('\\','/')
            replaces[f] = to_snake_case(f)
            # print(cur + '/' + f)

def walkdir2(dirname):
    for cur, _dirs, files in os.walk(dirname):
        pref = ''
        print(cur)
        # head, tail = os.path.split(cur)
        # while head:
        #     pref += '---'
        #     head, _tail = os.path.split(head)
        # print(pref+tail)
        for f in files:
            src = ''
            path = cur + '/' + f
            with open(path, 'r') as fd:
                src = fd.read()
                for r in replaces:
                    src = src.replace(r, replaces[r])
            if f == 'CMakeLists.txt':
                with open(path, 'w') as fd:
                    fd.write(src)
            else:
                os.remove(path)
                with open(cur + '/' + to_snake_case(f), 'w') as fd:
                    fd.write(src)

def to_snake_case(s:str):
    r = ''
    i = 0
    while i < len(s):
        if s[i].isupper():
            tmp = ''
            while i < len(s) and s[i].isupper():
                tmp += s[i].lower()
                i += 1
            if r:
                r += '_'
            r += tmp
        else:
            r += s[i]
            i += 1
    return r

walkdir1('include')
walkdir1('src')
walkdir2('include')
walkdir2('src')
# print(to_snake_case('EndPoint.cpp'))
# print(to_snake_case('SIMD.cpp'))
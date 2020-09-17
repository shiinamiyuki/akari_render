import sys

if __name__ == '__main__':
    src = sys.argv[1]
    out = sys.argv[2]
    name = sys.argv[3]
    out_buf = 'const char ' + name  + '[] = '
    with open(src, 'r') as f:
        data = f.read()
        out_buf += '{' + ','.join([hex(ord(c)) for c in data])
        out_buf += ', 0};\n'
    with open(out, 'w') as f:
        f.write(out_buf)
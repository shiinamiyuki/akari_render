import sys
import os
if __name__ == '__main__':
    out = sys.argv[1]
    sources = sys.argv[2:]
    gen = '// THIS FILE IS AUTO GENERATED\n'
    for src in sources:
        with open(src, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # if line.startswith('#include'):
                #     continue
                gen += line
        gen += '\n'
    with open(out, 'w') as f:
        f.write(gen + '\n')
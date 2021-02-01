import os
import sys
import argparse
if __name__ == '__main__':
    # files = sys.argv[1:-3]
    # assert '-o' == sys.argv[-3]
    # header_filename = sys.argv[-2]
    # source_filename = sys.argv[-1]

    parser = argparse.ArgumentParser()
    parser.add_argument('files',metavar='files',nargs='+',type=str)
    parser.add_argument('--ns', type=str)
    parser.add_argument('--header', type=str)
    parser.add_argument('--source', type=str)
    args = parser.parse_args()

    header_filename = args.header
    source_filename = args.source
    header = '#include <string_view>\n'
    source = '#include <string_view>\n'

    if args.ns:
        header += 'namespace ' + args.ns + '{\n'
        source += 'namespace ' + args.ns + '{\n'

    for filename in args.files:
        with open(filename, 'r') as f:
            root, ext = os.path.splitext(filename)
            head, tail = os.path.split(root)
            var_name = tail
            var_name = var_name.replace('-', '_').replace('.','_')
            header += 'extern const char ' + var_name + '[];\n'
            source += 'extern const char ' + var_name + '[]={' + ','.join(map(lambda x:str(ord(x)), f.read())) + '};\n'
    if args.ns:
        header += '}\n'
        source += '}\n'
    with open(header_filename, 'w') as f:
        f.write(header)
    with open(source_filename, 'w') as f:
        f.write(source)

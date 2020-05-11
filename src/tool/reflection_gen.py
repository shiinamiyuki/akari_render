import sys
import re
import os

'''
class:
    name: str
    methods: [ str ]
    fields: [ str ]

options:
    plugin_name
    
'''


def find_close_bracket(src: str, start: int):
    cnt = 0
    first = True
    for i in range(start, len(src)):
        if src[i] == '{':
            cnt += 1
            first = False
        if src[i] == '}':
            cnt -= 1
        if cnt == 0 and not first:
            return i + 1
    return None


def remove_comments(src: str):
    src = re.sub('//.*(\r\n?|\n)', '', src)
    src = re.sub('/\*(.|\r\n?|\n)\*/', '', src)
    src = src.replace('final', '')
    return src


def parse(src: str, options: dict):
    src = remove_comments(src)
    generated = dict()
    plugin_name = options['plugin_name']
    _identifier = r'[_A-Za-z][_0-9A-Za-z]*'
    _attribute = r'\[\[[^\]]+\]\]'
    _find_class = 'class[^{{]+{{'.format(_attribute, _identifier, _identifier)
    while True:
        class_ = re.search(_find_class, src)
        if class_ is None:
            break
        meta = {}
        meta['fields'] = []
        meta['methods'] = []
        # print('find ' + src[class_.start(): class_.end()])
        class_decl = src[class_.start(): class_.end()]
        end = find_close_bracket(src, class_.start())
        # print(classname)
        if ':' in class_decl:
            class_decl = re.split('(?<!:):(?!:)', class_decl)[0]
            classname = [x for x in class_decl.split() if x][-1]
        else:
            classname = [x for x in class_decl.split() if x][-2]
        # print('detected ' + classname)
        attribute_refl = re.compile(r'\[\[\s*(akari::refl|refl)\s*\]\]')
        field_refl = re.compile(r'\[\[\s*(akari::refl|refl)\s*\]\]((.|\r\n?|\n)*?)\s+((.|\r\n?|\n)*?);')
        if classname == plugin_name or attribute_refl.search(class_decl):
            print('generating meta info for ' + classname + ' ...')
            class_body = src[class_.start(): end]
            is_component = 'AKR_DECL_COMP' in class_body or 'AKR_IMPLS' in class_body
            bases = set()
            if is_component:
                bases.add('Component')
            if 'AKR_IMPLS' in class_body:
                impls = re.search(r'AKR_IMPLS\s*\(.*?\)', class_body)
                impls = class_body[impls.start():impls.end()]
                impls = impls[len('AKR_IMPLS'):].strip('(').strip(')')
                impls = [x.strip() for x in impls.split(',')]
                for impl in impls:
                    bases.add(impl)
            meta['bases'] = bases
            # print(class_body)
            start = 0
            while True:
                match = field_refl.search(class_body, start)
                if not match:
                    break
                field = class_body[match.span()[0]: match.span()[1] - 1].strip()
                # print(match.group(1))
                # print('field: ' + field)
                if '=' in field:
                    field = field.split('=')[0]
                field_sep = [x for x in field.split(' ') if x]
                field_name = field_sep[-1]
                # print(field_name)
                meta['fields'].append(field_name)
                start = match.end()
            generated[classname] = meta
            # constructor_ = re.compile(classname + r'\s*\((.|\r\n?|\n)*?\)\s*?[;{]')
            # start = 0
            # while True:
            #     match = constructor_.search(class_body, start)
            #     if not match:
            #         break
            #     ctor = class_body[match.span()[0]: match.span()[1] - 1].strip()
            #     args = ctor[len(classname):].strip().strip('(').strip(')')
            #     print(args)
            #     start = match.end()
        src = src[end:]
    # print(generated)
    generated_src = 'void _AkariGeneratedMeta(Plugin &p){\n'
    for classname in generated:
        meta = generated[classname]
        generated_src += ' {\n'
        bases = ''
        for base in list(meta['bases']):
            bases += ',' + base
        generated_src += '  auto c = class_<{}{}>("{}");\n'.format(classname, bases, classname)

        for field_name in meta['fields']:
            generated_src += '  c.property("{}", &{}::{});\n'.format(field_name, classname, field_name)

        generated_src += '  c.method("save", &{}::save);\n  c.method("load", &{}::load);\n'.format(classname, classname)

        generated_src += ' }\n'
    generated_src += '}\n'
    for classname in generated:
        meta = generated[classname]
        fields = ','.join(meta['fields'])
        generated_src += 'void {}::save(serialize::OutputArchive &ar)const{{\n'.format(classname)
        if meta['fields']:
            generated_src += '''  akari::serialize::AutoSaveVisitor v{{ar}};
                  _AKR_DETAIL_REFL(v, {});\n'''.format(fields)
        generated_src += '}\n'
        generated_src += 'void {}::load(serialize::InputArchive &ar){{\n'.format(classname)
        if meta['fields']:
            generated_src += '''  akari::serialize::AutoLoadVisitor v{{ar}};
                  _AKR_DETAIL_REFL(v, {});\n'''.format(fields)
        generated_src += '}\n'
    generated_src += '#define __AKR_PLUGIN_NAME__ "{}"'.format(plugin_name)
    # print(generated_src)
    return generated_src


# def test():
#     with open('../render/materials/Disney.cpp', 'r') as f:
#         options = {
#             'plugin_name': 'DisneyMaterial'
#         }
#         result = parse(f.read(), options)
#         # print(result)


# test()

# _attribute = r'\[\[[^\]]+\]\]'
# x = re.search(_attribute, 'class [[ attr]] A')
# print(x.start())
if __name__ == '__main__':
    target = sys.argv[1]
    main_file = sys.argv[2]
    file = sys.argv[3]
    with open(main_file, 'r') as f:
        options = {
            'plugin_name': target
        }
        result = parse(f.read(), options)
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'w') as f:
        f.write(result)

# drawn inspiration from pbrt-v4

import sys

"""
example:

{
    'flat':['Float'],
    'soa':{
        'Array<Float, 3>' : {
            'template':['Float']
            'fields':{
                'x()':'Float',
                'y()':'Float',
                'z()':'Float
            }
        }
    }
}
"""


"""
struct SOA<STRUCT> {
    SOA<Member> member ...;
};

"""
test_src = {
    'flat': ['Float', 'Point3f'],
    # 'defined':{
    #     'Array<Float, 3>' : {
    #         'template':['Float']
    #     }
    # },
    'soa': {
        # 'Point3f':{
        #     'template':[],
        #     'fields':{
        #         'x':'Float',
        #         'y':'Float',
        #         'z':'Float'
        #     }
        # },
        'Ray<C>': {
            'template': ['C'],
            'fields': {
                'o': 'Point3f',
                'd': 'Point3f',
                'tmin': 'Float',
                'tmax': 'Float'
            }
        }
    }
}
if __name__ == '__main__':
    src_file = sys.argv[1]
    with open(src_file, 'r') as f:
        src = f.read()
    # src = test_src
    cfg = eval(src)
    headers = cfg['headers']
    # cfg = test_src
    flats = set(cfg['flat'])
    soas = cfg['soa']

    def gen_struct(name, desc):
        indent = 4
        out = ''

        class Block:
            def __enter__(self):
                nonlocal indent
                indent += 4

            def __exit__(self, type, value, tb):
                nonlocal indent
                indent -= 4

        def wl(s):
            nonlocal out
            out += ' ' * indent + s + '\n'

        templates = desc['template']
        fields = desc['fields']
        if templates:
            wl("template<" +
               ','.join(['typename ' + i for i in templates]) + ">")
        wl('struct SOA<' + name + '>{')
        with Block():
            if 'Float' in templates:
                wl('AKR_IMPORT_CORE_TYPES();')
            elif 'C' in templates:
                wl('AKR_IMPORT_TYPES();')
            wl('size_t _size = 0;')
            for field in fields:
                assert fields[field] in flats or fields[field] in soas or fields[field] in templates, '{} is illegal'.format(
                    fields[field])
                wl('SOA<' + fields[field] + '> ' + field + ';')
            wl('using Self = SOA<' + name + '>;')
            wl('using value_type = ' + name + ';')
            wl('Self() = default;')
            wl('template<class Allocator>')
            wl('Self(size_t s, Allocator & alloc): _size(s)')
            for field in fields:
                wl(', ' + field + '(s, alloc)')
            wl('{}')
            wl('struct IndexHelper{')
            with Block():
                wl('Self & self;')
                wl('int idx;')
                wl('AKR_XPU operator value_type(){')
                with Block():
                    wl('value_type ret;')
                    for field in fields:
                        wl('ret.' + field + ' = self.' + field + '[idx];')
                    wl('return ret;')
                wl('}')
                wl('AKR_XPU const value_type & operator=(const value_type & rhs){')
                with Block():
                    for field in fields:
                        wl('self.' + field + '[idx] = rhs.' + field + ';')
                    wl('return rhs;')
                wl('}')
            wl('};')
            wl('struct ConstIndexHelper{')
            with Block():
                wl('const Self & self;')
                wl('int idx;')
                wl('AKR_XPU operator value_type(){')
                with Block():
                    wl('value_type ret;')
                    for field in fields:
                        wl('ret.' + field + ' = self.' + field + '[idx];')
                    wl('return ret;')
                wl('}')
            wl('};')
            wl('AKR_XPU IndexHelper operator[](int idx){return IndexHelper{*this, idx};}')
            wl('AKR_XPU ConstIndexHelper operator[](int idx)const{return ConstIndexHelper{*this, idx};}')
            wl('AKR_XPU size_t size()const{return _size;}')
            # wl('void resize(size_t s){')
            # with Block():
            #     wl('_size = s;')
            #     for field in fields:
            #         wl(field + '.resize(s);')
            # wl('}')
        wl('};')
        return out
    with open(sys.argv[2], 'w') as out_file:
        for h in headers:
            out_file.write('#pragma once\n#include <' + h + '>\n')
        out_file.write('namespace akari {\n')
        for soa in soas:
            out_file.write(gen_struct(soa, soas[soa]) + '\n')
        out_file.write('}\n')

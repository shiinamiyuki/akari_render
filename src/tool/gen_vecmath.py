def gen_basic_vec(ty:str, n: int):
    out = ''
    if n == 2:
        fields = 'xy'
    elif n == 3:
        fields = 'xyz'
    typename = ty + str(n)
    ctors = 
    out += 'struct {} {{{} {}; {}}};'.format(typename, ty, ','.join(fields),ctors)
    print(out)


gen_basic_vec('float',3)
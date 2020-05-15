import re


class Combinator:
    def __init__(self, f):
        self.f = f

    def reset(self, other):
        if isinstance(other, Combinator):
            self.f = other.f
        else:
            self.f = other

    def __call__(self, s, start):
        return self.f(s, start)

    def __or__(self, other):
        def f(s, start):
            m = self(s, start)
            if m is None:
                return other(s, start)
            m1 = other(s, start)
            if m1 is None:
                return m
            return max(m, m1)
        return Combinator(f)

    def __rshift__(self, other):
        def f(s, start):
            m = self(s, start)
            if m is not None:
                return other(s, m)
            return None
        return Combinator(f)

    def rep(self, min=0, max=99999):
        def f(s, start):
            prev = end = start
            for i in range(max):
                m = self(s, prev)
                if m is None and i >= min:
                    return prev
                if m is None:
                    return None
                end = m
                prev = end
            return end
        return Combinator(f)
        
    def opt(self):
        return self.rep(0,1)


def regex(r):
    p = re.compile(r)

    def f(s, start):
        m = p.match(s, start)
        if m is None:
            return None
        return m.end()

    return Combinator(f)


def lit(r):
    def f(s, start):
        if start == len(s) and r == '':
            return start
        if s.startswith(r, start):
            return len(r) + start
        return None

    return Combinator(f)


def place_holder():
    return Combinator(None)


def get_cpp_parser():
    identifier = regex(r'[_a-zA-Z][_a-zA-Z0-9]*')
    spc = regex(r'\s|\r\n|\n')
    spc_x = spc.rep()
    spc_opt = spc.opt()
    spc_1 = spc.rep(min=1)
    ns_name = (identifier >> spc_x >> lit("::") >> spc_x).rep() >> identifier
    qualifier = regex(r'const|volatile|mutable').opt()
    type_ = place_holder()
    type_decl = qualifier >> spc_x >> type_ >> (spc_x >> regex(r'\*|&|&&')).opt()#>> spc_x >> identifier >> \
    template_arg = Combinator(type_decl)
    template_instance = ns_name >> spc_x >> lit('<') \
        >> template_arg >> (spc_x >> lit(',') >> template_arg >> spc_x).rep() \
        >> spc_x >> lit('>')
    type_.reset(template_instance | ns_name)
    param_decl = type_decl >> spc_x >> identifier.opt()
    attribute = lit('[[') >> spc_x >> ns_name >> spc_x >> lit(']]')
    function_decl = attribute.opt() >> param_decl >> spc_x >> lit('(')
    field_decl = attribute.opt() >> spc_x >> param_decl >> spc_x >> (lit('=') | lit(';'))
    attr_type_decl = attribute.opt() >> spc_x >> type_decl
    parsers = {
        'identifier': identifier,
        'ns_name':ns_name,
        'type_decl':type_decl,
        'param_decl':param_decl,
        'template_instance':template_instance,
        'field_decl':field_decl,
        'function_decl':function_decl,
        'attr_type_decl':attr_type_decl,
        'type': type_
    }
    return parsers


# src = r'const std:: vector&& x'
# src = 'void compute_scattering_functions(SurfaceInteraction *si, MemoryArena &arena, TransportMode mode,'
# m = get_cpp_parser()['function_decl'](src, 0)
# assert m is not None
# print(m, src[:m])

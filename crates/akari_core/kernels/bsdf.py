from luisa_lang import lang as lc

def reflect(wo:lc.float3, n:lc.float3) -> lc.float3:
    z = wo - 2.0 * lc.dot(wo, n) * n
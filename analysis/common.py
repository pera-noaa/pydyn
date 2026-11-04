"""What the analysis scripts share."""


def parse_limits(spec):
    """A sub-cycle cap triple 'x,y,z' (Check_CFL's cfl_max_nloop_{x,y,z}) as ints."""
    lx, ly, lz = (int(v) for v in spec.split(","))
    return lx, ly, lz

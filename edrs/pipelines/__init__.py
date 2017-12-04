from .reduction import Reduction

def reduce_echelle(instrument):
    if instrument == 'FOCES':
        from .foces import FOCES
        reduction = FOCES()
        reduction.reduce()
    elif instrument == 'XinglongHRS':
        from .xl216hrs import XinglongHRS
        reduction = XinglongHRS()
        reduction.reduce()
    else:
        pass

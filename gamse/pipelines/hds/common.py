
def get_region_lst(header):
    """Get a list of array indices.

    Args:
        header ():
    Returns:
        tuple:
    """
    nx = header['NAXIS1']
    ny = header['NAXIS2']
    binx = header['BIN-FCT1']
    biny = header['BIN-FCT2']

    if (nx, ny)==(2148, 4100) and (binx, biny)==(1, 1):
        sci1_x1, sci1_x2 = 0, 1024
        sci1_y1, sci1_y2 = 0, 4100
        ovr1_x1, ovr1_x2 = 1024, 1024+50
        ovr1_y1, ovr1_y2 = 0, 4100

        sci2_x1, sci2_x2 = 1024+50*2, 1024*2+50*2
        sci2_y1, sci2_y2 = 0, 4100
        ovr2_x1, ovr2_x2 = 1024+50, 1024+50*2
        ovr2_y1, ovr2_y2 = 0, 4100

    elif (nx, ny)==(1124, 2050) and (binx, biny)==(2, 2):
        sci1_x1, sci1_x2 = 0, 512
        sci1_y1, sci1_y2 = 0, 2050
        ovr1_x1, ovr1_x2 = 512, 512+50
        ovr1_y1, ovr1_y2 = 0, 2050

        sci2_x1, sci2_x2 = 512+50*2, 512*2+50*2
        sci2_y1, sci2_y2 = 0, 2050
        ovr2_x1, ovr2_x2 = 512+50, 512+50*2
        ovr2_y1, ovr2_y2 = 0, 2050

    else:
        print(nx, ny, binx, biny)
        pass

    return [
            ((sci1_x1, sci1_x2, sci1_y1, sci1_y2),
             (ovr1_x1, ovr1_x2, ovr1_y1, ovr1_y2)),
            ((sci2_x1, sci2_x2, sci2_y1, sci2_y2),
             (ovr2_x1, ovr2_x2, ovr2_y1, ovr2_y2)),
            ]

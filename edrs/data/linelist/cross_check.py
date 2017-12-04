#!/usr/bin/env python3
import numpy as np
linelist1 = []
file1 = open('thar.dat')
for row in file1:
    row = row.strip()
    if len(row)==0 or row[0]=='#':
        continue
    wv = float(row)
    linelist1.append((wv, None))
file1.close()

linelist2 = []
file2 = open('thar-noao.dat')
for row in file2:
    row = row.strip()
    if len(row)==0 or row[0]=='#':
        continue
    g = row.split()
    wv      = float(g[0])
    if len(g)>1:
        species = g[1]
    else:
        species = None
    linelist2.append((wv, species))
file2.close()

for line in linelist1:
    wv = line[0]
    wvdiff = np.array([abs(wv-line[0]) for line in linelist2])
    if wvdiff.min()<1e-3:
        continue
    else:
        print('in thar but not in thar-noao:', wv)

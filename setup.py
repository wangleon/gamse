#!/usr/bin/env python3
from distutils.core import setup

setup(
    name         = 'gamse',
    version      = '0.93',
    description  = 'Data Reduction Software for High-resolution Echelle Spectrographs',
    author       = 'Liang Wang',
    author_email = 'lwang@mpe.mpg.de',
    license      = 'BSD',
    scripts      = ['scripts/gamse',
                    ],
    packages     = [
                    'gamse',
                    'gamse/utils',
                    'gamse/echelle',
                    'gamse/pipelines',
                    'gamse/pipelines/foces',
                    'gamse/pipelines/xinglong216hrs',
                   ],
    package_data = {
                    'gamse': ['data/config/*',
                              'data/linelist/*',]
                    },
    )

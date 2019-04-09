#!/usr/bin/env python3
from distutils.core import setup

setup(
    name         = 'gamse',
    version      = '0.9',
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
                   ],
    package_data = {
                    'gamse': ['data/config/*',
                              'data/linelist/*',]
                    },
    )

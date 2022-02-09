#!/usr/bin/env python3
from distutils.core import setup
from gamse import __version__

setup(
    name         = 'gamse',
    version      = __version__,
    description  = 'Data Reduction Software for High-resolution Echelle Spectrographs',
    author       = 'Liang Wang',
    author_email = 'lwang@mpe.mpg.de',
    license      = 'Apache-2.0',
    scripts      = ['scripts/gamse',
                    ],
    packages     = [
                    'gamse',
                    'gamse/utils',
                    'gamse/echelle',
                    'gamse/pipelines',
                    'gamse/pipelines/espadons',
                    'gamse/pipelines/feros',
                    'gamse/pipelines/foces',
                    'gamse/pipelines/harps',
                    'gamse/pipelines/hds',
                    'gamse/pipelines/hires',
                    'gamse/pipelines/levy',
                    'gamse/pipelines/lhrs',
                    'gamse/pipelines/uves',
                    'gamse/pipelines/xinglong216hrs',
                   ],
    package_data = {
                    'gamse': ['data/calib/*',
                              'data/config/*',
                              'data/linelist/*',
                              ]
                    },
    )

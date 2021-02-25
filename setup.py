#!/usr/bin/env python3
from distutils.core import setup
from gamse import __version__

setup(
    name         = 'gamse',
    version      = __version__,
    description  = 'Data Reduction Software for High-resolution Echelle Spectrographs',
    author       = 'Liang Wang',
    author_email = 'lwang@mpe.mpg.de',
    license      = 'BSD',
    scripts      = ['scripts/gamse',
                    'scripts/gamse-show',
                    ],
    packages     = [
                    'gamse',
                    'gamse/utils',
                    'gamse/echelle',
                    'gamse/pipelines',
                    'gamse/pipelines/feros',
                    'gamse/pipelines/foces',
                    'gamse/pipelines/hds',
                    'gamse/pipelines/hires',
                    'gamse/pipelines/levy',
                    'gamse/pipelines/lhrs',
                    'gamse/pipelines/xinglong216hrs',
                   ],
    package_data = {
                    'gamse': ['data/config/*',
                              'data/linelist/*',]
                    },
    )

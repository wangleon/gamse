#!/usr/bin/env python3
from distutils.core import setup

setup(
    name         = 'gamse',
    version      = '0.98',
    description  = 'Data Reduction Software for High-resolution Echelle Spectrographs',
    author       = 'Liang Wang',
    author_email = 'wang.leon@gmail.com',
    license      = 'Apache-2.0',
    zip_safe     = False,
    scripts      = ['scripts/gamse',
                    'scripts/gamse_ident',
                    ],
    packages     = [
                    'gamse',
                    'gamse/utils',
                    'gamse/echelle',
                    'gamse/pipelines',
                    'gamse/pipelines/espadons',
                    'gamse/pipelines/espresso',
                    'gamse/pipelines/feros',
                    'gamse/pipelines/foces',
                    'gamse/pipelines/harps',
                    'gamse/pipelines/hds',
                    'gamse/pipelines/hires',
                    'gamse/pipelines/levy',
                    'gamse/pipelines/lhrs',
                    'gamse/pipelines/uves',
                    'gamse/pipelines/xinglong216hrs',
                    'gamse/pipelines/lires',
                   ],
    package_data = {
                    'gamse': ['data/calib/*',
                              'data/config/*',
                              'data/linelist/*',
                              ]
                    },
    )

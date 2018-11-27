#!/usr/bin/env python3
from distutils.core import setup

setup(
    name         = 'edrs',
    version      = '2.0',
    description  = 'Echelle Data Reduction Software',
    author       = 'Liang Wang',
    author_email = 'lwang@mpe.mpg.de',
    license      = 'BSD',
    scripts      = ['scripts/edrs2',
                    ],
    packages     = [
                    'edrs',
                    'edrs/utils',
                    'edrs/echelle',
                    'edrs/pipelines',
                   ],
    package_data = {
                    'edrs': ['data/config/*',
                             'data/linelist/*',]
                    },
    )

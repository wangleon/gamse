#!/usr/bin/env python
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
                    'edrs/ccdproc',
                    'edrs/echelle',
                    'edrs/pipelines',
                   ],
    package_data = {
                    'edrs': ['data/linelist/*',]
                    },
    )


GAMSE
=====
<img src="https://github.com/wangleon/gamse/blob/master/docs/images/gamse.svg" width=250>

[![PyPI](https://img.shields.io/pypi/v/gamse.svg)](https://pypi.org/project/gamse/)
[![Read the Docs](https://img.shields.io/readthedocs/gamse.svg)](https://gamse.readthedocs.io/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

`GAMSE` is a data reduction package for high-resolution échelle spectrographs.
It contains necessary subroutines in spectral reduction process, including
overscan correction, bias subtraction, order detection, flat-fielding
correction, background correction, and optimal extraction.

Dependencies
------------
`GAMSE` is based on Python 3.4 or later, and does not work in Python 2.x.
To use `GAMSE` the following packages are required:

* [Numpy](http://www.numpy.org/) 1.16.1 or later: A Python library for
  multi-dimensional arrays and mathematics.
* [Scipy](https://www.scipy.org/) 0.17.0 or later: A Python library for
  scientific computing.
* [Matplotlib](https://matplotlib.org/) 2.2.0 or later: To display and generate
  output figures.
* [Astropy](http://www.astropy.org/) 3.1.1 or later: To read and write FITS
  files and ASCII tables.

Installation
------------
To install `GAMSE` package with `pip`, simply use the following command:

```bash
sudo pip install gamse
```

Or alternatively, clone the whole repository with GIT:

```bash
git clone https://github.com/wangleon/gamse.git
```

Then run the setup script in the cloned directory:

```bash
sudo python3 setup.py install
```



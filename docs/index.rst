
GAMSE Documentation
===================

.. image:: images/gamse.svg
    :alt: GAMSE logo
    :width: 250px

`GAMSE` is a data reduction package for high-resolution échelle spectrographs.
It contains necessary subroutines in spectral reduction process, including
overscan correction, bias subtraction, order detection, flat-fielding
correction, background correction, and optimal extraction.
GAMSE supports a variety of instruments, e.g. FOCES on the 2m Fraunhofer
Telescope in Wendelstein Observatory, and HRS on the 2.16m telescope in Xinglong
Observatory.

Installation
------------

To install `GAMSE`, simply use `pip`

.. code-block:: bash

   sudo pip install gamse

Or alternatively, use git

.. code-block:: bash

   git clone https://github.com/wangleon/gamse.git

then run the following command in the `GAMSE` directory

.. code-block:: bash

   sudo python3 setup.py install

Introduction
-------------

Modes of execution 
-------------------
This software package is designed to be easily adapted the data taken with an other echelle spectrograph.

The software has three modii namely:
    * **normal**
        Mode for the normal day to day use.
    * **learn**
        Mode to teach the code to reduce data from other echelle spectrographs then FOCES.
    * **compare**
        This mode compare the results obtained with this data reduction pipline to other reduction piplines.


Supported Spectrographs
-------------------------
* :ref:`FOCES<pipeline_foces>` on 2m Fraunhofer Telescope in Wendelstein Observatory
* :ref:`HRS<pipeline_xinglong216hrs>` on 2.16m telescope in Xinglong Observatory
* :ref:`Levy<>` on APF
.. * :ref:`HIRES<pipeline_hires>` on 10m Keck II Telescope in  W. M. Keck Observatory

Steps performed by the software
--------------------------------
The pipline performs the correction and wavelength calibration of the science data (therfore reduces the 2d image taken by the echelle spectrograph to a 1d list containing the normalisied photons over wavelength data). To do so the following tasks are performed:

    1. Overscan correction
    2. Bias subtraction
    3. Dark subtraction
    4. Order tracing
    5. Background subtraction
    6. Flatfield correction
    7. Wavelength calibration


See also
--------
* :ref:`Structure of output FITS files<fits_output>`


Indices and Tables
--------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


References
--------------
.. [#Pfeiffer1998] Pfeiffer et al., 1998, *A&AS*, 130, 381 :ads:`1998A&AS..130..381P`
.. [#Steinmetz2008] Steinmetz et al., 2008, *Science*, 321, 1335 :ads:`2008Sci...321.1335S`
.. [#Wilken2012] Wilken et al., 2012, *Nature*, 485, 611 :ads:`2012Natur.485..611W`

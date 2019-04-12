.. _manual_foces:

Reduction Manual for FOCES
==========================


Introduction to the Instrument
------------------------------
Fibre Optics Cassegrain Echelle Spectrograph (FOCES, Pfeiffer et al. 1998
[#Pfeiffer1998]_) was originally mounted to the Cassegrain focus of the 2.2m
telescope in Calar Alto Observatory, Spain.
After 15 years of operation, the spectrograph was brought back to `University
Obserevatory Munich (USM) <http://www.usm.uni-muenchen.de/>`_ for a major
upgrade (Grupp et al. 2009 [#Grupp2009]_, 2010 [#Grupp2010]_) to meet the
requirements for precise spectroscopy, such as searching for extra-solar
planets with the Doppler method.
In the summer of 2017, FOCES was successfully mounted on the 2m Fraunhofer
Telescope (Hopp et al. 2014 [#Hopp2014]_) at the `Wendelstein Observatory
<http://www.wendelstein-observatorium.de:8002/wst_en.html>`_ in Southern
Bavaria, Germany.
FOCES is connected to a Nasmyth focus of the telescope via an octagonal
multi-mode optical fiber.
FOCES has a resolving power (*R*) of ~ 70,000, and covers the wavelength range
of 390 - 900 nm.

The characteristics of FOCES are summaried as below:

.. list-table::
   :widths: 7 10

   * - **Spectral resolving power**
     - *R* = *λ*\ /Δ\ *λ* = 70,000
   * - **Wavelength coverage**
     - 390 - 900 nm
   * - **Connection to the telescope**
     - octagonal multi-mode fiber
   * - **Main-disperser**
     - 31.6 lines mm\ :sup:`−1` R2 VPHG
   * - **Cross-disperser**
     - A pair of prisms
   * - **Detector**
     - 2048 x 2048 pixels
   * - **Pixel size**
     - 13.5 μm
   * - **Sampling per resolution element**
     - ~2.3 pixels
   * - **Wavelength calibration**
     - ThAr/astrocomb simultaneous reference

Preparing the Data
------------------
The first step is to create a new working directory in a place where you have
full read/write permissions.
All the steps performed by `GAMSE` will be done in this directory.
For example, the following commands create a new working directory called
``foces.2018-07-18``, where the FOCES data taken in July 18, 2018 will be
reduced here.

.. code-block:: bash

   mkdir foces.2018-07-18
   cd foces.2018-07-18

Then, a text file containing the necessary information is required to tell
`GAMSE` which instrument the data is obtained with, and the path to raw data.
The name of the text file is arbitrary, but the suffix must be ``.cfg``.
The user must make sure there is only one ``.cfg`` file in the working
directory.
For example, a text file called ``foces-2018-07-18.cfg`` with the following
contents is created:
::

    [data]
    telescope  = Fraunhofer
    instrument = FOCES
    rawdata    = rawdata

With the first two keywords `GAMSE` will call the FOCES pipeline to reduce the
data.
The third keyword ``rawdata`` tells the software the path to the raw images.
The default value is a sub-directory called ``rawdata`` in the working
directory.
The user may want to keep the raw data in their original places, but to use a
softlink to the actual data path, instead.
For example, the raw images taken in July 18, 2018 are in
``/data/foces/rawdata/2018/0718/``, and the following command is to create a
softlink called ``rawdata`` in the working directory:

.. code-block:: bash

   ln -s /data/foces/rawdata/2018/0718 rawdata

Or alternatively, one can use the actual data path in the configuration file:
::

    [data]
    telescope  = Fraunhofer
    instrument = FOCES
    rawdata    = /data/foces/rawdata/2018/0718

In this case, the soft link to the data path is not necessary any more.

Generating the Observing Log
----------------------------
The following command scans all the FITS files in the data path as specified
in the configure file:

.. code-block:: bash

   gamse list

``GAMSE`` will extract some information from the FITS files and print an
observing log as a table in the terminal:
::

    ------ ------------------------------ ------- ------------ ------- ----------------------- ------- ------
    frameid             fileid             imgtype    object    exptime         obsdate          nsat    q95  
    ------- ------------------------------ ------- ------------ ------- ----------------------- ------- ------
          0 20180718_0001_FOC1800_SCI0       sci   Unknown           20 2018-07-18T20:55:47.000       0  23228
    ... ...
          0 20180718_0017_FOC1800_THA1       cal   ThAr             1.5 2018-07-18T21:44:26.000   13503   1069
          0 20180718_0018_FOC1800_SCI0       sci   Unknown          180 2018-07-18T22:23:37.000       0    943
    ... ...
          0 20180719_0012_FOC1800_THA1       cal   ThAr             1.5 2018-07-19T01:43:28.000   13305   1066
          0 20180719_0013_FOC1800_THA2       cal   ThAr               3 2018-07-19T01:45:35.000   23582   1241
          0 20180719_0014_FOC1800_FLA1       cal   Flat             1.5 2018-07-19T01:50:33.000       9  21349
          0 20180719_0023_FOC1800_FLA1       cal   Flat             1.5 2018-07-19T02:06:39.000      84  21687
          0 20180719_0024_FOC1800_FLA1       cal   Flat             1.5 2018-07-19T02:08:15.000      92  21701
    ... ...
          0 20180719_0025_FOC1800_FLA2       cal   Flat               6 2018-07-19T02:10:22.000  338893  64638
          0 20180719_0026_FOC1800_FLA2       cal   Flat               6 2018-07-19T02:12:03.000  339258  64640
          0 20180719_0027_FOC1800_FLA2       cal   Flat               6 2018-07-19T02:13:49.000  339597  64638
    ... ...
          0 20180719_0035_FOC1800_BIA0       cal   Bias            0.01 2018-07-19T02:31:27.000       0    908
          0 20180719_0036_FOC1800_BIA0       cal   Bias            0.01 2018-07-19T02:33:08.000       0    908
    ... ...
    ------- ------------------------------ ------- ------------ ------- ----------------------- ------- ------

Menwhile, a text file with the name of ``2018-07-18.obslog`` containing almost
the same table will be created in the working directory.
The columns have the explicit meanings as shown in the header.
``nsat`` is the number of saturated pixels of the whole image, and ``q95`` is
the 95% quantile value of all pixels.
The values of these two columns are extracted from the FITS images, and the
others are taken from the FITS headers or generated automatically (``frameid``
and ``imgtype``).
See :ref:`Observing Log <obslog>` for more details about this table.

Since the target names of FOCES are not written into the headers of FITS files,
User has to open the obslog file with a text editor, and make some changes
*manually*.

The obslog files will *NOT* be overwritten by running ``gamse list``, but new
files named ``2018-07-18.1.oblog``, ``2018-07-18.2.oblog``... with extra numbers
will be generated if there are existing obslog files in the working directory.
Users have to decide which observing log file to use in the data reduction.



The FOCES pipeline is used to reduced the raw FITS files generated by FOCES.
The naming of the FITS files follows the :ref:`FOCES naming convention<foces_fitsname>` and
the keywords in the primary header follows the :ref:`FOCES FITS standard<foces_fitsfile>`\ .
The procedure of the reduction of FOCES data includes:

#. Creation of log & configuration files;
#. Overscan correction;
#. Bias subraction;
#. Dark current correction;
#. Flat fielding correction;
#. Order location;
#. Background subtraction;
#. 1-D spectra extraction;
#. Wavelength calibration.



Log & Config Files
------------------
The list of different type of exposures (flat, bias, science...) is given in the
configuration file (`*****.cfg`).
Users can also change the parameters used in EDRS2 (e.g. degree of polynomials
in the background correction, scanning intervals in order location...).
Below is an example::

    [reduction]
    path.data    = rawdata
    path.midproc = midproc
    path.report  = report
    path.result  = result

    bias    = 2-6
    thar    = 1, 47, 48
    flat_1  = 43
    flat_2  = 44
    flat_3  = 45
    flat_4  = 46
    science = 7-42

    overscan.variation_fig = overscan_variation.png

    bias.cosmic_clip     = 10.
    bias.bias_file       = bias.fits
    bias.smooth_sigma    = 3
    bias.smooth_mode     = nearest
    bias.smooth_file     = bias_smooth.fits
    bias.res_file        = bias_res.fits
    bias.variation_fig   = bias_change.png
    bias.smooth_fig      = bias_smooth.png
    bias.smooth_hist_fig = bias_smooth_hist.png

    flat.flat_file   = flat.fits
    flat.mosaic_file = flat.reg

    trace.trace_file = trace.fits

The configuration file follows the
`RFC822 <https://tools.ietf.org/html/rfc822.html>`_ format, and composed of
several "sessions" marked with `[XXX]`.
It is read by Python built-in
`ConfigParser <https://docs.python.org/2/library/configparser.html>`_ module and
passed to EDRS2.

Usage of FOCES Pipeline
-----------------------

Generation of Observing Log
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The first step is to create a reduction directory and create link to the raw
data in this directory.
For example, the raw images (`***.fits`) are saved in `~/data/foces/2015-03-04`,
then the command is:

.. code-block:: bash

    ln -s ~/data/foces/2015-03-04 rawdata

Then, run the following command to generate the obseving log file.
EDRS2 will print all items on the screen.

.. code-block:: bash

    edrs2 list rawdata

To generate an observing log file, just redirect the output to a specific file:

.. code-block:: bash

    edrs2 list rawdata > 2015-03-04.log

Check the Config File
^^^^^^^^^^^^^^^^^^^^^
Make sure there is a configuration file (`XXX.cfg`) in the reduction directory.

The config file must have a `reduction` section and should contain the following
options::

    [reduction]
    path.data    = rawdata
    path.midproc = midproc
    path.report  = report
    path.result  = result

Overscan
^^^^^^^^


Flat Fielding
^^^^^^^^^^^^^

Flats with different exposure times are mosaiced together to generate the final
flat image.
In EDRS2, an interactive interface will be displayed

.. figure:: ../images/flat_mosaic.png
   :align: center
    
The cross sections of each kind of flat are plotted with different colors.
Then user can simply click the figure to define a mosaic boundary line,
and select which part of the flat is used in the final flat (the lowest one with
black color).
Because the order in the 2D image is not a straight line, the boundary line on
the 2D image lies between two orders, and is carefully calculated to fit the
curvatures of the orders, to avoid any crossing, as shown below:

.. figure:: ../images/flat_mosaic2.png
   :align: center
    
The mosaic module can be used not only 3 kinds of flat (red/green/blue), but any
numbers of kinds.
This step only relies on matplotlib, scipy, and DS9.

Logging
--------
During running, a log file is generated to tell the users what did the program
do, and where is the error occurred if the program quit abnormally.
The log file `edrs.log` is generated in the current working directory by the
Python `logging <https://docs.python.org/2/library/logging.html>`_ module,
and has a clear and machine-readable format, and list the time, module name,
line number of the running place, and the name of function.
As below::

    * 2016-02-29T11:15:39.511 [INFO] __main__ - 67 - record_system_info():
      Start reduction.
      Node:              wangliang-mbp
      Processor:         1 x Intel(R) Core(TM) i7-2620M CPU @ 2.70GHz (2 cores)
      System:            Linux 3.19.0-51-generic x86_64
      Distribution:      Ubuntu 14.04 trusty
      Memory:            7.7G (total); 4.2G (used); 3.5G (free)
      Username:          wangliang
      Python version:    2.7.6
      Working directory: /home/wangliang/work/foces/reduction/2015-03-04
    --------------------------------------------------------------------------------
    * 2016-02-29T11:15:39.511 [INFO] __main__ - 77 - main():
      arg1 = foces, start reducing FOCES data
    --------------------------------------------------------------------------------
    * 2016-02-29T11:15:39.511 [INFO] edrs.config - 36 - read_config():
      Found config file: "FOCES_20150304_A.cfg"
    --------------------------------------------------------------------------------
    * 2016-02-29T11:15:39.512 [ERROR] edrs.pipeline.foces.reduce_data - 18 - reduce_data():
      data_path: "rawdata" does not exist
    --------------------------------------------------------------------------------

References
-----------
.. [#Grupp2009] Grupp et al., 2009, *SPIE*, 7440, 74401G :ads:`2009SPIE.7440E..1GG`
.. [#Grupp2010] Grupp et al., 2010, *SPIE*, 7735, 773573 :ads:`2010SPIE.7735E..73G`
.. [#Hopp2014] Hopp et al., 2014, *SPIE*, 9145, 91452D :ads:`2014SPIE.9145E..2DH`
.. [#Pfeiffer1998] Pfeiffer et al., 1998, *A&AS*, 130, 381 :ads:`1998A%26AS..130..381P`

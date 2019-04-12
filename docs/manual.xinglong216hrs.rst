.. |e| replace:: e\ :sup:`−`

.. _manual_xinglong216hrs:

Reduction Manual for Xinglong 2.16m HRS
=======================================

Introduction to Instrument
--------------------------
Since 2009, A fiber-fed High Resolution Spectrograph (HRS) was attached on the
Cassegrain focus of the 2.16m telescope in Xinglong Observatory of National
Astronomical Observatories, Chinese Academy of Sciences (CAS).
The spectrograph has a resolving power (*R* = *λ*/Δ\ *λ*) of ~49,800 at the slit
width of 0.19 mm, covering the wavelength range of 365 ~ 1000 nm.
The CCD detector is a back-illuminated E2V CCD 203-82 chip with 4096 × 4096
pixels and the pixel size of 12.0 μm.
For more information, see Fan et al. 2016 [#Fan2016]_.

.. csv-table::
   :header: Parameter, Value
   :widths: 6, 10

   Wavelength Coverage,       "3,650 - 10,000 Å"
   Resolving Power,           "32,000 - 106,000"
   Fiber Diameter,            2".4/1".6
   CCD Detector,              "E2V 4k × 4k back illuminated, 12 μm/pixel"
   Radial Velocity Precision, ±6 m/s (with an I\ :sub:`2` cell)


CCD Detector
^^^^^^^^^^^^

The CCD gain and readout noise of four gates at different readout speeds are
summarized as below:

+---------------+-----------+---------------------------------------------------------------+-------------+-------------+--------------+-------------+
| Readout Speed | Gain Mode | Gain (|e|/ADU)                                                | Read Out Noise (|e|/pixel)                             |
|               |           +---------------+---------------+---------------+---------------+-------------+-------------+--------------+-------------+
|               |           | L0            | L1            | R0            | R1            | L0          | L1          |  R0          | R1          |
+===============+===========+===============+===============+===============+===============+=============+=============+==============+=============+
| Slow          | 0         | 2.369 ± 0.146 | 2.309 ± 0.067 | 2.928 ± 0.314 | 2.358 ± 0.102 | 3.89 ± 0.31 | 3.86 ± 0.32 |  6.88 ± 2.26 | 4.03 ± 0.27 |
|               +-----------+---------------+---------------+---------------+---------------+-------------+-------------+--------------+-------------+
|               | 1         | 1.005 ± 0.023 | 1.022 ± 0.020 | 1.074 ± 0.031 | 1.033 ± 0.019 | 2.84 ± 0.13 | 2.84 ± 0.15 |  3.38 ± 0.38 | 2.90 ± 0.11 |
|               +-----------+---------------+---------------+---------------+---------------+-------------+-------------+--------------+-------------+
|               | 2         | 0.430 ± 0.004 | 0.447 ± 0.007 | 0.448 ± 0.004 | 0.448 ± 0.006 | 2.47 ± 0.07 | 2.54 ± 0.06 |  2.72 ± 0.08 | 2.58 ± 0.12 |
+---------------+-----------+---------------+---------------+---------------+---------------+-------------+-------------+--------------+-------------+
| Medium        | 0         | 4.954 ± 0.566 | 4.590 ± 0.196 | 8.381 ± 0.000 | 5.096 ± 0.622 | 6.89 ± 1.03 | 6.38 ± 0.55 | 15.34 ± 5.20 | 7.19 ± 1.13 |
|               +-----------+---------------+---------------+---------------+---------------+-------------+-------------+--------------+-------------+
|               | 1         | 2.301 ± 0.163 | 2.189 ± 0.065 | 2.847 ± 0.303 | 2.303 ± 0.137 | 4.60 ± 0.46 | 4.29 ± 0.21 | 6.58 ± 1.18  | 4.61 ± 0.37 | 
|               +-----------+---------------+---------------+---------------+---------------+-------------+-------------+--------------+-------------+
|               | 2         | 0.863 ± 0.009 | 0.896 ± 0.007 | 0.937 ± 0.003 | 0.891 ± 0.009 | 3.20 ± 0.12 | 3.22 ± 0.07 | 3.69 ± 0.11  | 3.29 ± 0.09 |
+---------------+-----------+---------------+---------------+---------------+---------------+-------------+-------------+--------------+-------------+
| Fast          | 0         | 2.406 ± 0.140 | 2.484 ± 0.090 | 2.866 ± 0.192 | 2.521 ± 0.186 | 9.31 ± 0.74 | 8.52 ± 0.70 | 19.37 ± 9.26 | 9.02 ± 0.77 |
|               +-----------+---------------+---------------+---------------+---------------+-------------+-------------+--------------+-------------+
|               | 1         | 1.010 ± 0.021 | 1.057 ± 0.020 | 1.075 ± 0.039 | 1.027 ± 0.029 | 7.88 ± 0.40 | 7.33 ± 0.44 | 16.69 ± 6.92 | 7.34 ± 0.52 |
|               +-----------+---------------+---------------+---------------+---------------+-------------+-------------+--------------+-------------+
|               | 2         | 0.431 ± 0.008 | 0.451 ± 0.018 | 0.463 ± 0.004 | 0.448 ± 0.004 | 6.84 ± 0.40 | 6.27 ± 0.41 | 8.28 ± 0.26  | 6.62 ± 0.38 |
+---------------+-----------+---------------+---------------+---------------+---------------+-------------+-------------+--------------+-------------+


Spectral Reduction
------------------

Generation of Observing Log
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following command scans the raw images and generate an observing log file
with name of `YYYY-MM-DD.obslog`, where `YYYY-MM-DD` is the date of the *first*
FITS image in the data folder.

.. code-block:: bash

   $ edrs2 list

If the file name already exists, `YYYY-MM-DD.1.obslog`, `YYYY-MM-DD.2.obslog`
... will be used as substituions.

At this step, the program also reads a file `obsinfo.txt` if it exists.
This file contains addtional infomations, such as whether an iodine cell is
used, and the accurate starting time of an exposure, since these parameters are
not included in the FITS files of HRS on the Xinglong 2.16m telescope.
An example of `obsinfo.txt` is shown below::

        frameid  object   i2cell      obsdate       
          int      str     bool         time        
        -------- -------- ------ -------------------
        01-10    Flat                               
        12-21    Flat                               
        22       I2                                 
        23       thar                               
        24       thar                               
        25       HD195820 True   2014-11-03T18:25:27
        26       HD195820 True   2014-11-03T19:00:44
        27       HD210460 False  2014-11-03T19:28:04
        ... ...

comparing to the 'ascii.fixed_width_two_line' format in `astropy.table` module,
the second row contains the data types of table columns.


Perpare the Configuration File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The pipeline accepts the following options in the config file:

.. csv-table:: Accepted options in config file
   :header: Section, Option, Type, Default Value, Description
   :escape: '
   :widths: 10, 18, 10, 18, 60

   **data**,       **telescope**,     *str*,    Xinglong216,     Name of the telescope (Fixed).
   **data**,       **instrument**,    *str*,    HRS,             Name of the instrument (Fixed).
   **data**,       **rawdata**,       *str*,    rawdata,         Path to the rawdata.
   **reduce**,     **midproc**,       *str*,    midproc,         Path to the mid-process folder. Will be created if not exist.
   **reduce**,     **result**,        *str*,    result,          Path to the result 1-D spectra folder. Will be created if not exist.
   **reduce**,     **report**,        *str*,    report,          Path to the report folder. Will be created if not exist.
   **reduce**,     **mode**,          *str*,    normal,          "Reduction mode. Available modes include '"normal'", '"debug'" and '"fast'"."
   **bias**,       **cosmic_clip**,   *float*,  10,              Upper clipping value for removing the cosmic-rays.
   **bias**,       **maxiter**,       *int*,    5,               Maximum iteration numbers.
   **bias**,       **smooth_method**, *str*,    ,                Method of smoothing the bias data.
   **bias**,       **smooth_sigma**,  *int*,    ,                Sigma of Gaussian smoothing core.
   **bias**,       **smooth_mode**,   *str*,    ,                Mode of Gaussian smoothing core.
   **trace**,      **scan_step**,     *int*,    ,                Steps of pixels used to scan along the main dispersion direction.
   **trace**,      **minimum**,       *float*,  ,                Minimum value to filter the input image.
   **trace**,      **seperation**,    *float*,  ,                Estimated order seperations (in pixel) at *y* = 0 along the cross-dispersion.
   **trace**,      **sep_der**,       *float*,  ,                Estimated first derivative of seperations per 1000 pixels along the *y* axis.
   **trace**,      **filling**,       *float*,  ,                Fraction of detected pixels to total step of scanning.
   **trace**,      **display**,       *bool*,   ,                Display a figure on screen if *yes*.
   **trace**,      **degree**,        *int*,    ,                Degree of polynomial used to describe the positions of orders.
   **background**, **scan_step**,     *int*,    ,                Steps of pixels used to scan along the main dispersion direction.
   **background**, **xorder**,        *int*,    ,                Degree of 2D polynomial along *x*-axis (dispersion direction).
   **background**, **yorder**,        *int*,    ,                Degree of 2D polynomial along *y*-axis (cross-dispersion direction).
   **background**, **maxiter**,       *int*,    ,                Maximum number of iteration of 2D polynomial fitting.
   **background**, **upper_clip**,    *float*,  ,                Upper sigma clipping threshold.
   **background**, **lower_clip**,    *float*,  ,                Lower sigma clipping threshold.
   **background**, **extend**,        *bool*,   ,                Extend the grid to the whole image if *True*.
   **background**, **display**,       *bool*,   ,                Display a graphics if *yes*.

References
----------
.. [#Fan2016] Fan et al., 2016, *PASP*, 128, 115005 :ads:`2016PASP..128k5005F`

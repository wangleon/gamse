
.. _config:

Config File
===========
`GAMSE` uses configuration files to control the input parameters and the
procedures performed during the data reduction.
A configuration file is a plain text file with the name of ``XXX.cfg``.
It follows the `RFC 822 <https://tools.ietf.org/html/rfc822.html>`_ format
and is similiar to ``.INI`` files in Microsoft Windows systems.
Briefly, it consists of a series of sections led by headers like ``[XXX]``, and
each section contains some entries like ``name: value`` or ``name=value``.
Comments prefixed by hash (``#``) or semicolons (``;``) symbols are supported,
for example:

.. code-block:: ini

   [data]
   telescope   = Fraunhofer
   instrument  = FOCES
   rawdata     = rawdata   # path to raw data
   statime_key = FRAME     # key of starting time of exposure in FITS header
   exptime_key = EXPOSURE  # key of exposure time in FITS header
   direction   = xb+

   [reduce]
   midproc     = midproc   # path to mid process files
   report      = report    # path to reports
   result      = onedspec  # path to one dimensional spectra
   mode        = normal
   fig_format  = png

   [reduce.bias]
   bias_file     = ${reduce:midproc}/bias.fits
   cosmic_clip   = 10      # upper clip threshold
   maxiter       = 5       # maximum iteration number

   [reduce.trace]
   minimum    = 8
   scan_step  = 100
   separation = 500:26, 1500:15
   filling    = 0.3
   align_deg  = 2
   display    = no
   degree     = 3
   ... ...

``GAMSE`` includes built-in configuration files for some spectrographs.
Therefore, users do not have to find these values their own.
The above example is the content of the built-in configuration file for FOCES
spectrograph.

Sometimes, one may want to change the values of a few entries.
In these cases, users can create a new ``.cfg`` file in the working directory
and write these entries together with their section names in.
Only the entries in the new configuration files are overridden during the data
reduction.

For example, a user-created ``.cfg`` file with:

.. code-block:: ini

   [data]
   telescope   = Fraunhofer
   instrument  = FOCES
   rawdata     = rawdata

   [reduce]
   fig_format  = pdf

will only override the value of ``fig_format`` in the ``reduce`` section in the
built-in configuration files.

.. _config_entries:

List of Accepted Entries
------------------------
`Gamse` uses a set of general keywords in the config file to descrip the data
formats.
They are in the ``[data]`` section of the config files.

.. list-table:: Accepted entries in ``[data]`` section
   :widths: 18, 10, 18, 60
   :header-rows: 1

   * - Key
     - Type
     - Default Value
     - Description
   * - **telescope**
     - *str*
     - Fraunhofer
     - Name of the telescope (Fixed).
   * - **instrument**
     - *str*
     - FOCES
     - Name of the instrument (Fixed).
   * - **rawdata**
     - *str*
     - rawdata
     - Path to the rawdata.
   * - **statime_key**
     - *str*
     - FRAME
     - Key of starting time of exposure in FITS header.
   * - **exptime_key**
     - *str*
     - EXPOSURE
     - key of exposure time in FITS header.
   * - **direction**
     - **str**
     -
     - Direction of the echelle spectrum on the CCD.

* **telescope** and **instrument**: Name of the telescope and the instrument.

* **rawdata**: Path to the folder for the raw data.
  It tells the software where to find the raw images.
  The default value is a sub-directory called ``rawdata`` in the working
  directory.
  The user may want to keep the raw data in their original places, but to use a
  soft link to the actual data path, instead.
  For example, the raw images taken on July 18, 2018, are in
  ``/data/foces/rawdata/2018/0718/``, and the following command is to create a
  soft link called ``rawdata`` in the working directory:

  .. code-block:: bash

     ln -s /data/foces/rawdata/2018/0718 rawdata

  Alternatively, one can use the actual data path in the configuration file:

  .. code-block:: ini

    rawdata    = /data/foces/rawdata/2018/0718

  In this case, the soft link to the data path is not necessary anymore.

* **statime_key**: Key of the starting time in the FITS header.

* **exptime_key**: Key of the exposure time inf the FITS header.

* **direction**: Direction of the Echelle spectrum on the CCD.
  A typical direction string is composed of three letters, like ``xb+`` or
  ``yr-``, where
 
  * The first letter indicates the axes of main-dispersion direction (either
    ``x`` or ``y``).
  * The second letter, either ``b`` or ``r`` is the direction of red/blue
    orders. 
    ``b`` means the blue orders locates in the smaller row (if the first letter
    is ``x``) or column (if the first letter is ``y``) numbers in the CCD.
    And ``r`` means the red orders vice-versa.
  * The last letter indicates whehter the wavelength is increasing (in this
    case, ``+``) or descreasing (in this case, ``-``) along the increasing pixel
    number within an échelle order.

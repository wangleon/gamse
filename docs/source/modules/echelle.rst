Echelle Module
==============

CCD Image Processing
--------------------
.. currentmodule:: edrs.imageproc
.. autosummary::
    combine_fits
    savitzky_golay_2d
    array_to_table
    table_to_array

.. automodule:: edrs.imageproc
   :members:
   :private-members:
   :undoc-members:


Order Location and Tracing
--------------------------
.. currentmodule:: edrs.echelle.trace
.. autosummary::
    ApertureLocation
    ApertureSet
    find_apertures
    load_aperture_set

.. automodule:: edrs.echelle.trace
   :members:
   :private-members:
   :undoc-members:

Flat Fielding Correction
------------------------
.. currentmodule:: edrs.echelle.flat
.. autosummary::
    load_mosaic
    mosaic_flat_auto
    mosaic_flat_interact
    save_mosaic_reg
    detect_gap
    get_slitfunc
    get_flatfielding

.. automodule:: edrs.echelle.flat
   :members:
   :private-members:
   :undoc-members:

Background Correction
---------------------
.. currentmodule:: edrs.echelle.background
.. autosummary::
    correct_background

.. automodule:: edrs.echelle.background
   :members:
   :private-members:
   :undoc-members:

1-D Spectra Extraction
----------------------
.. currentmodule:: edrs.echelle.extract
.. autosummary::
    sum_extract

.. automodule:: edrs.echelle.extract
   :members:
   :private-members:
   :undoc-members:

Wavelength Calibration
----------------------
.. currentmodule:: edrs.echelle.wvcalib
.. autosummary::
    CalibWindow
    CalibFigure
    PlotFrame
    CustomToolbar
    InfoFrame
    LineTable
    FitparaFrame
    wvcalib
    recalib
    fit_wv
    get_wv_val
    guess_wavelength
    find_order
    find_drift
    find_local_peak
    find_shift_ccf
    find_shift_ccf2
    load_ident
    save_ident
    is_identified
    load_linelist
    search_linelist
    reference_wv
    get_aperture_coeffs_in_header

.. automodule:: edrs.echelle.wvcalib
   :members:
   :private-members:
   :undoc-members:

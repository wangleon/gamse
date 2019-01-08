Echelle Module
==============

CCD Image Processing
--------------------
.. currentmodule:: edrs.echelle.imageproc
.. autosummary::
    combine_images
    savitzky_golay_2d
    array_to_table
    table_to_array
    fix_pixels

.. automodule:: edrs.echelle.imageproc
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
    load_aperture_set_from_header
    gaussian_bkg
    fitfunc
    errfunc

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
    mosaic_images
    save_mosaic_reg
    detect_gap
    get_fiber_flat
    get_slit_flat
    default_smooth_aperpar_A
    default_smooth_aperpar_k
    default_smooth_aperpar_c
    default_smooth_aperpar_bkg

.. automodule:: edrs.echelle.flat
   :members:
   :private-members:
   :undoc-members:

Background Correction
---------------------
.. currentmodule:: edrs.echelle.background
.. autosummary::
    find_background
    fit_background
    interpolate_background

.. automodule:: edrs.echelle.background
   :members:
   :private-members:
   :undoc-members:

1-D Spectra Extraction
----------------------
.. currentmodule:: edrs.echelle.extract
.. autosummary::
    sum_extract
    extract_aperset

.. automodule:: edrs.echelle.extract
   :members:
   :private-members:
   :undoc-members:

Wavelength Calibration
----------------------
calibration functions:

.. currentmodule:: edrs.echelle.wlcalib
.. autosummary::
    wlcalib
    recalib
    fit_wavelength
    get_wavelength
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
    reference_wl
    reference_wl_new
    get_aperture_coeffs_in_header

GUI-related:

.. currentmodule:: edrs.echelle.wlcalib
.. autosummary::
    CalibWindow
    CalibFigure
    PlotFrame
    CustomToolbar
    InfoFrame
    LineTable
    FitparaFrame

.. automodule:: edrs.echelle.wlcalib
   :members:
   :private-members:
   :undoc-members:

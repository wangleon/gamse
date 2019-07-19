.. _wlcalib:

Wavelength Calibration
======================

The wavelength of the spectra needs to be calibrated.
The most common way of doing such this calibration for an echelle spectrograph
is to compare the positions of science spectra with that of a hollow cathode
lamp with known wavelengths, e.g., the Thorium-Argon (ThAr) lamp or Uranium-Neon
(UNe) lamp.
The wavelengths (*λ*) for any pixel (*x*) in a given order (*m*) can be built by
fitting a 2-D polynomial from the positons of spectral features, of which the
wavelengths have been precisely determined.

In `GAMSE`, there are several modes of wavelength calibration.
They are designed to fulfill the various requirements.
These are summaried as below.

* For a spectrograph with unkown echelle formats, `GAMSE` will display a
  graphical user interface (GUI), which helps the users to identify the spectral
  lines by hand. 
* In the case that there are more than one frames with the same hollow cathode
  lamp under the same configuration (e.g., resolution, CCD binning), users can
  identify the wavelengths by hand for only one of them, and all the others will
  be identified automatically taking the references of the manually-identified
  spectra, suppose that the positions of the spectra on the detector have only
  small amounts of drift.
* If the camera or the detector rotated by 180° or the order positions move by a
  large amount of pixels on the detector over a long-term, `GAMSE` is still be
  able to find the correct echelle order number (*m*) and identify the
  wavelengths automatically.



APIs
----
calibration functions:

.. currentmodule:: gamse.echelle.wlcalib
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
    select_calib_from_database

GUI-related:

.. currentmodule:: gamse.echelle.wlcalib
.. autosummary::
    CalibWindow
    CalibFigure
    PlotFrame
    CustomToolbar
    InfoFrame
    LineTable
    FitparaFrame


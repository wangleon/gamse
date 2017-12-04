.. _foces_fitsname:


FOCES Naming Convention
========================

The FOCES naming convention for observational files is::

    YYYYMMDD_NNNN_XXXYYNN_XXXN.fits

Where

* `YYYYMMDD` is the year, month and date like `20150730` for July 30th. 2015;
* `NNNN` is the unique daily expossure number from 0001 to 9999;
* `XXXYYNN`:

  * `XXX` is the instrument descriptor (`FOC` means FOCES);
  * `YYNN` is the year and revision. e.g. `1501` for setup 01 2015. Revision counts up with hardware changes, such as new fibers, changed alignments, new calibration lamps.

* `XXXN`: Exposure type and descriptor.

  * `FLA` means FlatField, `0` means Blue, `1` means Green, and `2` means Red;
  * `DAR` means Dark
  * `WAV` means Wavelength calibrator. N= `1` means ThAr, N= `2` means UNe
  * `SCI` means Science target. N= `0` means Star alone, N= `1` means Single fiber cal, N= `2` means Double fiber cal, and N= `3` means Iodine cell cal

Below lists the intrument codes:

* `FOC` means FOCES
* `BOE` means BOES (TBC)
* `W1M` means Weihai 1m echelle spectrograph
* `X2M` means Xinglong 2m echelle spectrograph

See also
----------
* :ref:`pipeline_foces`
* :ref:`foces_fitsfile`

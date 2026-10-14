import dateutil.parser
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from ...utils.onedarray import iterative_savgol_filter
from ...echelle.wlcalib import (wlcalib, recalib,
                                get_calib_weight_lst, find_caliblamp_offset,
                                get_calib_from_header,
                                reference_spec_wavelength,
                                reference_self_wavelength,
                                )
from ...echelle.extract import extract_aperset, extract_aperset_optimal
from ..engine import FrameStep, resolve_reference, FrameResult
from ..base import ImageFrame, SpectrumFrame
from ..common import get_spectype
from .dataframe import read_dataframe

class OverscanSubtractionStep(FrameStep):
    def run(self, result, context, **options):
        # get dataframe from input result
        dataframe = result.frame
        data = dataframe.data
        mask = dataframe.mask

        data, mask = correct_overscan(data, mask, **options)

        extra_head = dataframe.extra_head.copy()
        prefix = 'HIERARCH REDUCTION OVERSCAN '
        extra_head.append((prefix + 'CORRECTED', True))

        new_dataframe = ImageFrame(data = data,
                                   head = dataframe.head,
                                   mask = mask,
                                   extra_head = extra_head,
                                   )
        new_result = FrameResult(frame = new_dataframe)
        return new_result
        

class ExtractionStep(FrameStep):
    def run(self, result, context, **options):
        product = resolve_reference(options['input'], context)
        aperset = product.value

        lower_limit = 15
        upper_limit = 15

        dataframe = result.frame
        data = dataframe.data
        mask = dataframe.mask

        spectra1d = extract_aperset(data, mask,
                                    apertureset = aperset,
                                    lower_limit = lower_limit,
                                    upper_limit = upper_limit,
                                    )

        ######### initialize spectype
        ny, nx = dataframe.data.shape
        if aperset[0].direct == 0:
            ndisp = ny  # dispersion direction is Y
        elif aperset[0].direct == 1:
            ndisp = nx  # dispersion direction is X
        else:
            raise ValueError

        spectype = get_spectype(ndisp)

        # pack to a structured array
        spec = []
        for aper, item in sorted(spectra1d.items()):
            flux_sum = item['flux_sum']
            n = flux_sum.size
            xloc = item['x']
            yloc = item['y']

            # pack to table
            row = (aper, 0,                         # aperture and order number
                    xloc,                           # x
                    yloc,                           # y
                    np.zeros(n, dtype=np.float64),  # wavelength
                    flux_sum,                       # flux
                    np.zeros(n, dtype=np.float32),  # error
                    np.zeros(n, dtype=np.float32),  # background
                    np.zeros(n, dtype=np.int16),    # mask
                    )
            spec.append(row)
        spec = np.array(spec, dtype=spectype)

        extra_head = dataframe.extra_head.copy()
        prefix = 'HIERARCH REDUCTION EXTRACTION '
        extra_head.append((prefix + 'METHOD', 'SUM'))
        extra_head.append((prefix + 'LOWERLIM', lower_limit))
        extra_head.append((prefix + 'UPPERLIM', upper_limit))

        specframe = SpectrumFrame(data = spec,
                                  head = dataframe.head,
                                  extra_head = extra_head,
                                  )
        print('  - Extraction Finished. {} orders extracted'.format(len(spec)))
        new_result = FrameResult(frame = specframe)

        return new_result


class CalibrateWavelengthStep(FrameStep):
    def run(self, result, context, **options):
        print("Calibrate Wavelength")

        dataframe = result.frame

        # get reference spec and calib
        filepath = 'new.fits'
        hdu_lst = fits.open(filepath)
        head = hdu_lst[0].header
        spec = hdu_lst[1].data
        hdu_lst.close()

        calib = get_calib_from_header(head)
        ref_path = filepath
        ref_spec = spec
        ref_calib = calib

        ref_direction = ref_calib['direction']

        # self.parent is the FrameEngine
        # self.parent.parent is the PipelineStep
        # self.parent.parent is the Pipeline
        pipeline = self.parent.parent.parent
        instrument = pipeline.instrument
        direction = instrument.direction

        if direction[1] == '?':
            aperture_k = None
        elif direction[1] == ref_direction[1]:
            aperture_k = 1
        else:
            aperture_k = -1

        if direction[2] == '?':
            pixel_k = None
        elif direction[2] == ref_direction[2]:
            pixel_k = 1
        else:
            pixel_k = -1

        _result = find_caliblamp_offset(ref_spec, dataframe.data,
                                        aperture_k       = aperture_k,
                                        pixel_k          = pixel_k,
                                        pixel_range      = (-30, 30),
                                        max_order_offset = 10,
                                        mode             = 'debug',
                                       )
        aperture_koffset = (_result[0], float(_result[1]))
        pixel_koffset    = (_result[2], float(_result[3]))

        use_prev_fitpar = False
        xorder      = None if use_prev_fitpar else 4
        yorder      = None if use_prev_fitpar else 3
        maxiter     = None if use_prev_fitpar else 5
        clipping    = None if use_prev_fitpar else 3
        window_size = None if use_prev_fitpar else 11
        q_threshold = None if use_prev_fitpar else 10

        wlfit_filter = lambda item: item['pixel'] < 4200
        
        calib, fig = recalib(dataframe.data,
                             ref_spec         = ref_spec,
                             linelist         = 'ThAr',
                             aperture_koffset = aperture_koffset,
                             pixel_koffset    = pixel_koffset,
                             ref_calib        = ref_calib,
                             xorder           = xorder,
                             yorder           = yorder,
                             maxiter          = maxiter,
                             clipping         = clipping,
                             window_size      = window_size,
                             q_threshold      = q_threshold,
                             direction        = direction,
                             fit_filter       = wlfit_filter,
                             )

        fileid   = dataframe.head['FILENAME']
        date_obs = dataframe.head['DATE-OBS']
        utc_obs  = dataframe.head['UTC-OBS']
        exptime  = dataframe.head['EXPTIME']
        # get obsdate
        obsdate_str = date_obs + 'T' + utc_obs
        obsdt = dateutil.parser.parse(obsdate_str)
        obsdate = obsdt.isoformat()[0:23]

        
        title = '{}.fits'.format(fileid)
        fig.suptitle(title)
        figname = 'wlcalib_{}.png'.format(fileid)
        figpath = context.figure_path / figname
        fig.savefig(figpath)
        plt.close(fig)
        print('wlcalib figure saved to ', figpath)

        # add more infos in calib
        calib['fileid']  = fileid
        calib['obsdate'] = obsdate
        calib['exptime'] = exptime
        
        # reference the ThAr spectra
        spec, card_lst, identlist = reference_self_wavelength(
                                        dataframe.data, calib)

        extra_head = dataframe.extra_head.copy()
        prefix = 'HIERARCH REDUCTION WLCALIB '
        for key, value in card_lst:
            extra_head.append((prefix+key, value))

        spec_frame = SpectrumFrame(data = spec,
                                   head = dataframe.head,
                                   extra_head = extra_head,
                                   ident_lst = identlist,
                                   )
        new_result = FrameResult(frame = spec_frame)
        new_result['calib'] = calib
        return new_result

class ApplyWavelengthStep(FrameStep):
    def run(self, dataframe, context, **options):

        print("Apply Wavelength")


def correct_overscan(data, mask, **kwargs):
    """Correct overscan.

    Args:
        data ():
        header ():
    Returns:

    """
    ny, nx = data.shape

    winlen = kwargs.get('winlen', 501)

    mean1 = data[:,0:20].mean(axis=1)
    # extent the mean array by winlen in the left and right
    mean1_ext = np.zeros((mean1.size+2*winlen),dtype=mean1.dtype)
    # copy the original mean array
    mean1_ext[winlen:winlen+mean1.size] = mean1
    mean1_ext[0:winlen] = mean1[0:winlen][::-1]
    mean1_ext[mean1.size+winlen:] = mean1[-winlen:][::-1]
    ovr1,_,_,_ = iterative_savgol_filter(mean1_ext,
                    winlen=winlen, order=3, upper_clip=3)
    ovr1 = ovr1[winlen:winlen+mean1.size]

    mean2 = data[:,nx-20:nx].mean(axis=1)
    mean2_ext = np.zeros((mean2.size+2*winlen),dtype=mean1.dtype)
    mean2_ext[winlen:winlen+mean2.size] = mean2
    mean2_ext[0:winlen] = mean2[0:winlen][::-1]
    mean2_ext[mean2.size+winlen:] = mean2[-winlen:][::-1]
    ovr2,_,_,_ = iterative_savgol_filter(mean2_ext,
                    winlen=winlen, order=3, upper_clip=3)
    ovr2 = ovr2[winlen:winlen+mean1.size]


    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot(mean1, lw=0.6, alpha=0.5)
    ax1.plot(ovr1, lw=0.6)

    ax2.plot(mean2, lw=0.6, alpha=0.6)
    ax2.plot(ovr2, lw=0.6)

    ax1.set_xlim(0,ny-1)
    ax2.set_xlim(0,ny-1)
    fig.savefig('{}_ovr.png'.format(fileid))
    plt.close(fig)
    '''

    scidata1 = data[:,20:nx//2]
    scidata2 = data[:,nx//2:nx-20]

    ovrimage1 = np.repeat([ovr1], scidata1.shape[1], axis=0).T
    ovrimage2 = np.repeat([ovr2], scidata2.shape[1], axis=0).T

    ovrdata = np.zeros((ny, nx-40), dtype=np.float64)
    ny1, nx1 = ovrdata.shape
    ovrdata[:, 0:nx1//2]   = scidata1 - ovrimage1
    ovrdata[:, nx1//2:nx1] = scidata2 - ovrimage2

    mask = mask[:,20:nx-20]

    #if verbose:
    #    print('Overscan', ovr1.mean(), ovr2.mean())

    return ovrdata, mask

import numpy as np
import astropy.io.fits as fits
from ...utils.onedarray import iterative_savgol_filter
from ...echelle.wlcalib import (wlcalib, recalib,
                                get_calib_weight_lst, find_caliblamp_offset,
                                get_calib_from_header,
                                reference_spec_wavelength,
                                reference_self_wavelength,
                                )
from ...echelle.extract import extract_aperset, extract_aperset_optimal
from ..engine import FrameStep, resolve_reference
from .dataframe import ESPADONSFrame

class OverscanSubtractionStep(FrameStep):
    def run(self, dataframe, context, **options):
        data, mask = correct_overscan(dataframe.data, dataframe.mask, **options)
        info = dataframe.info.copy()
        return ESPADONSFrame(data = data,
                             head = dataframe.head,
                             mask = mask,
                             info = info,
                             )

class BiasSubtractionStep(FrameStep):
    def run(self, dataframe, context, **options):
        bias_frame = resolve_reference(options['input'], context)
        data = dataframe.data - bias_frame.data
        info = dataframe.info.copy()
        return ESPADONSFrame(data = data,
                             head = dataframe.head,
                             mask = dataframe.mask,
                             info = info,
                             )

class FlatCorrectionStep(FrameStep):
    def run(self, dataframe, context, **options):
        print("This is a flat correction for a single frame")

class ExtractionStep(FrameStep):
    def run(self, dataframe, context, **options):
        aperset = resolve_reference(options['input'], context)

        # extract ThAr spectra
        spectra1d = extract_aperset(
                dataframe.data,
                dataframe.mask,
                apertureset = aperset,
                lower_limit = 15,
                upper_limit = 15,
                )

        ######### initialize spectype
        ny, nx = dataframe.data.shape
        if aperset[0].direct == 0:
            ndisp = ny  # dispersion direction is Y
        elif aperset[0].direct == 1:
            ndisp = nx  # dispersion direction is X
        else:
            raise ValueError

        types = [
                ('aperture',   np.int16),
                ('order',      np.int16),
                ('x',          (np.float32, ndisp)),
                ('y',          (np.float32, ndisp)),
                ('wavelength', (np.float64, ndisp)),
                ('flux',       (np.float32, ndisp)),
                ('error',      (np.float32, ndisp)),
                ('background', (np.float32, ndisp)),
                ('mask',       (np.int32,   ndisp)),
                ]
        names, formats = list(zip(*types))
        spectype = np.dtype({'names': names, 'formats': formats})


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

        return spec


class CalibrateWavelengthStep(FrameStep):
    def run(self, spec, context, **options):
        print("Calibrate Wavelength")
        
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
        print(self.parent.instrument.direction)

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

        result = find_caliblamp_offset(self.ref_spec, spec,
                                       aperture_k       = aperture_k,
                                       pixel_k          = pixel_k,
                                       pixel_range      = (-30, 30),
                                       max_order_offset = 10,
                                       mode             = 'debug',
                                      )
        aperture_koffset = (result[0], float(result[1]))
        pixel_koffset    = (result[2], float(result[3]))

        use_prev_fitpar = False
        xorder      = None if use_prev_fitpar else 4
        yorder      = None if use_prev_fitpar else 3
        maxiter     = None if use_prev_fitpar else 5
        clipping    = None if use_prev_fitpar else 3
        window_size = None if use_prev_fitpar else 11
        q_threshold = None if use_prev_fitpar else 10

        wlfit_filter = lambda item: item['pixel'] < 4200
        
        calib, fig = recalib(spec,
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

        
        #title = '{}.fits'.format(fileid)
        #fig.suptitle(title)
        #figname = 'wlcalib_{}.png'.format(fileid)
        figname = 'wlcalib_xxx.png'
        figpath = context.figure_path / figname
        fig.savefig(figpath)
        plt.close(fig)
        print('wlcalib figure saved to ', figpath)

        # add more infos in calib
        #calib['fileid']  = fileid
        #calib['obsdate'] = obsdate
        #calib['exptime'] = exptime
        
        # reference the ThAr spectra
        spec, card_lst, identlist = reference_self_wavelength(spec, calib)

        head = fits.Header() # temporary
        prefix = 'HIERARCH GAMSE WLCALIB '
        for key, value in card_lst:
            head.append((prefix+key, value))

        hdu_lst = fits.HDUList([
                    fits.PrimaryHDU(header=head),
                    fits.BinTableHDU(spec),
                    fits.BinTableHDU(identlist),
                    ])

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

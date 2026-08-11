import os
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
                                select_calib_auto,
                                )
from ...echelle.extract import extract_aperset, extract_aperset_optimal
from ..engine import FrameStep, resolve_reference, FrameResult
from ..base import ImageFrame, SpectrumFrame

from .common import correct_overscan, select_calib_from_database

class OverscanSubtractionStep(FrameStep):
    def run(self, result, context, **options):
        # get dataframe from input result
        dataframe = result.frame
        data    = dataframe.data
        binning = dataframe.extra_head['LOGINFO BINNING']
        amp     = dataframe.extra_head['LOGINFO AMP']

        newdata, newmask = correct_overscan(data, binning, amp)

        # append important information to new extra_head
        extra_head = dataframe.extra_head.copy()
        prefix = 'HIERARCH REDUCTION OVERSCAN '
        extra_head.append((prefix + 'CORRECTED', True))

        new_dataframe = ImageFrame(data = newdata,
                                   head = dataframe.head,
                                   mask = newmask,
                                   extra_head = extra_head,
                                   )
        new_result = FrameResult(frame = new_dataframe)
        return new_result


class BiasSubtractionStep(FrameStep):
    def run(self, result, context, **options):

        dataframe = result.frame

        # get bias from from context
        bias_product = resolve_reference(options['input'], context)
        bias_frame = bias_product.value

        data = dataframe.data - bias_frame.data

        extra_head = dataframe.extra_head.copy()
        prefix = 'HIERARCH REDUCTION BIAS '
        extra_head.append((prefix+'CORRECTED', True))
        extra_head.append((prefix+'MEAN', bias_frame.data.mean()))
        extra_head.append((prefix+'MEDIAN', np.median(bias_frame.data)))

        new_dataframe = ImageFrame(data = data,
                          head = dataframe.head,
                          mask = dataframe.mask,
                          extra_head = extra_head,
                          )
        new_result = FrameResult(frame = new_dataframe)
        return new_result

class FlatCorrectionStep(FrameStep):
    def run(self, result, context, **options):
        sens_product = resolve_reference(options['input'], context)
        sens_frame = sens_product.value

        dataframe = result.frame

        data = dataframe.data / sens_frame.data

        extra_head = dataframe.extra_head.copy()
        prefix = 'HIERARCH FLAT '
        extra_head.append((prefix+'CORRECTED', True))

        new_dataframe = ImageFrame(data = data,
                                   head = dataframe.head,
                                   mask = dataframe.mask,
                                   extra_head = extra_head,
                                   )
        new_result = FrameResult(frame = new_dataframe)

        return new_result

class ExtractionStep(FrameStep):
    def run(self, result, context, **options):
        product = resolve_reference(options['input'], context)
        aperset = product.value
        lower_limit = 7
        upper_limit = 7

        dataframe = result.frame

        spectra1d = extract_aperset(
                dataframe.data,
                dataframe.mask,
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

        extra_head = dataframe.extra_head.copy()
        prefix = 'HIERARCH REDUCTION '
        extra_head.append((prefix + 'METHOD', 'SUM'))
        extra_head.append((prefix + 'LOWERLIM', lower_limit))
        extra_head.append((prefix + 'UPPERLIM', upper_limit))

        specframe = SpectrumFrame(data = spec,
                                  head = dataframe.head,
                                  extra_head = extra_head,
                                  )
        print('Extraction finished')
        new_result = FrameResult(frame = specframe)

        return new_result
        
class CalibrateWavelengthStep(FrameStep):
    def run(self, result, context, **options):

        dataframe = result.frame

        # search for database
        index_file = os.path.join(os.path.dirname(__file__),
                        '../../data/calib/wlcalib_xinglong216hrs.dat')
        obsdate = dataframe.extra_head['LOGINFO OBSDATE']
        ref_spec, ref_calib = select_calib_from_database(
                            index_file, obsdate)

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
                                       pixel_range      = (-50, 50),
                                       max_order_offset = 15,
                                       mode             = 'debug',
                                       )
        aperture_koffset = (_result[0], float(_result[1]))
        pixel_koffset    = (_result[2], float(_result[3]))
        print(aperture_koffset, pixel_koffset)

        use_prev_fitpar = False
        xorder      = None if use_prev_fitpar else 4
        yorder      = None if use_prev_fitpar else 4
        maxiter     = None if use_prev_fitpar else 5
        clipping    = None if use_prev_fitpar else 3
        window_size = None if use_prev_fitpar else 13
        q_threshold = None if use_prev_fitpar else 10

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
                             )

        fileid   = dataframe.extra_head['LOGINFO FILEID']
        obsdate  = dataframe.extra_head['LOGINFO OBSDATE']
        exptime  = dataframe.extra_head['LOGINFO EXPTIME']

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

    def run(self, result, context, **options):

        print("Apply Wavelength")

        wave_product = resolve_reference(options['input'], context)
        calib_lst = wave_product.value

        dataframe = result.frame
        spec   = dataframe.data
        fileid = dataframe.extra_head['LOGINFO FILEID']

        rms_threshold    = 0.005
        group_contiguous = True
        time_diff        = 120

        ref_calib_lst = select_calib_auto(calib_lst,
                                          rms_threshold    = rms_threshold,
                                          group_contiguous = group_contiguous,
                                          time_diff        = time_diff,
                                          )
        ref_fileid_lst = [calib['fileid'] for calib in ref_calib_lst]

        # print ThAr summary and selected calib
        fmt_string = ' [{:3d}] {} - ({:4g} sec) - {:4d}/{:4d} RMS = {:7.5f}'
        for frameid, calib in sorted(calib_lst.items()):
            string = fmt_string.format(frameid, calib['fileid'],
                        calib['exptime'], calib['nuse'], calib['ntot'],
                        calib['std'])
            if calib['fileid'] in ref_fileid_lst:
                string = '\033[91m{} [selected]\033[0m'.format(string)
            print(string)

        obsdate = dataframe.extra_head['LOGINFO OBSDATE']
        exptime = dataframe.extra_head['LOGINFO EXPTIME']

        # wavelength calibration
        weight_lst = get_calib_weight_lst(ref_calib_lst,
                                          obsdate = obsdate,
                                          exptime = exptime,
                                          )

        message_lst = ['Wavelength calibration:']
        for i, calib in enumerate(ref_calib_lst):
            string = '{} ({:4g} sec) {} weight = {:5.3f}'.format(
                        calib['fileid'], calib['exptime'], calib['obsdate'],
                        weight_lst[i])
            message_lst.append(string)
        message = os.linesep.join(message_lst)
        print(message)

        spec, card_lst = reference_spec_wavelength(spec,
                            ref_calib_lst, weight_lst)

        extra_head = dataframe.extra_head.copy()
        prefix = 'HIERARCH REDUCTION WLCALIB '
        for key, value in card_lst:
            extra_head.append((prefix + key, value))

        specframe = SpectrumFrame(data = spec,
                                  head = dataframe.head,
                                  extra_head = extra_head,
                                  )
        filepath = context.onedspec_path / 'spec_{}.fits'.format(fileid)
        specframe.save(filepath, overwrite=True)
        print('spec saved to', filepath)
        new_result = FrameResult(frame = specframe)

        return new_result




def correct_overscan(data, binning, amp):

    naxis1 = data.shape[1]      # size along X axis
    naxis2 = data.shape[0]      # size along Y axis

    x1 = 0      # X origin
    y1 = 0      # Y origin

    xbin = int(binning.split('x')[0])
    ybin = int(binning.split('x')[1])

    # total pixels along Y and X axis
    ny, nx = 4136, 4096
    cover = naxis1 - nx//xbin
    rover = 0

    if amp in ['LTB', 'LBRT']:
        # 2 regions vertically
        # science & overscan region
        sci1 = (y1, y1+(naxis2-rover)//2, x1, x1+(naxis1-cover))
        ovr1 = (y1, y1+(naxis2-rover)//2, x1+(naxis1-cover), naxis1)

        sci2 = (y1+(naxis2-rover)//2, naxis2-rover, x1, x1+(naxis1-cover))
        ovr2 = (y1+(naxis2-rover)//2, naxis2-rover, x1+(naxis1-cover), naxis1)

        region_lst = ((sci1, ovr1), (sci2, ovr2))

        # get the size of the ENTIRE sci region
        y2 = y1 + (naxis2-rover)
        x2 = x1 + (naxis1-cover)

        # initialize the data after overscan
        newdata = np.zeros((y2-y1, x2-x1), dtype=np.float64)

        for iregion, (sci_region, ovr_region) in enumerate(region_lst):

            sci_y1, sci_y2, sci_x1, sci_x2 = sci_region
            ovr_y1, ovr_y2, ovr_x1, ovr_x2 = ovr_region

            scidata = data[sci_y1:sci_y2, sci_x1:sci_x2]
            ovrdata = data[ovr_y1:ovr_y2, ovr_x1+2:ovr_x2]

            # find the overscan level along the y-axis
            ovr_lst = ovrdata.mean(axis=1)

            # apply the sav-gol fitler to the mean of overscan
            winlen = 301
            order = 3
            upper_clip = 3.0
            ovr_smooth, _, _, _ = iterative_savgol_filter(ovr_lst,
                            winlen=winlen, order=order, upper_clip=upper_clip)
            
            # expand the 1d overscan values to 2D image that fits the sci region
            nysci = scidata.shape[1]
            ovrimg = np.repeat(ovr_smooth, nysci).reshape(-1, nysci)
            cordata = scidata - ovrimg
            
            new_y1 = sci_y1 - y1
            new_y2 = sci_y2 - y1
            new_x1 = sci_x1 - x1
            new_x2 = sci_x2 - x1
            
            newdata[new_y1:new_y2, new_x1:new_x2] = cordata
    
    else:
        # other amlifiers
        raise ValueError

    newmask = np.zeros_like(newdata, dtype=np.int16)

    return newdata, newmask

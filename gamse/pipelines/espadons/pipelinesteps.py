import os
import numpy as np
import astropy.io.fits as fits
from ...echelle.imageproc import combine_images, savitzky_golay_2d
from ..engine import (CollectionPipelineStep, AnalysisPipelineStep,
                       StreamingPipelineStep)
from .dataframe import ESPADONSFrame
from .common import (select_calib_from_database,
                    BackgroundFigure, SpatialProfileFigure)
from .trace import find_apertures
from .flat import (get_flat2, smooth_aperpar_A, smooth_aperpar_c,
                   smooth_aperpar_bkg)

class ProcessBias(CollectionPipelineStep):
    def finish(self, results, context, inputs, **options):
        print('Process Bias')

        n_bias = len(results)
        bias_data_lst = np.array([dataframe.data for dataframe in results])
        
        mode         = options.get('mode', 'mean')
        cosmic_clip  = options.get('cosmic_clip', 10)
        maxiter      = options.get('maxiter', 5)
        maskmode     = 'max' if n_bias>=3 else None

        # determine number of cores to be used
        ncores = os.cpu_count()

        bias_combine = combine_images(
                bias_data_lst,
                mode        = mode,
                upper_clip  = cosmic_clip,
                maxiter     = maxiter,
                maskmode    = maskmode,
                ncores      = ncores,
                )

        info = []
        info.append(('BIAS NFILE', n_bias))
        for iframe, dataframe in enumerate(results):
            key1   = 'BIAS FILEID {:03d}'.format(iframe+1)
            value1 = dataframe.head['FILENAME']
            info.append((key1, value1))
        info.append(('BIAS COMBINE_MODE', mode))
        info.append(('BIAS COSMIC_CLIP', cosmic_clip))
        info.append(('BIAS MAXITER', maxiter))

        bias_frame = ESPADONSFrame(
                        data = bias_combine,
                        head = fits.Header(),
                        mask = np.zeros_like(bias_combine, dtype=np.int16),
                        info = info,
                        )

        filename = options.get('file', None)
        if filename:
            filepath = context.midproc_path / filename
            bias_frame.save(filepath, overwrite=True)
            print('Bias saved to ', filepath)

        context[self.name] = {
                'bias': bias_frame
                }

class ProcessFlat(CollectionPipelineStep):
    def finish(self, results, context, inputs, **options):
        print('Process Flat')

        n_flat = len(results)
        flat_data_lst = np.array([dataframe.data for dataframe in results])

        mode         = options.get('mode', 'mean')
        cosmic_clip  = options.get('cosmic_clip', 10)
        maxiter      = options.get('maxiter', 5)
        maskmode     = 'max' if n_flat>=3 else None

        # determine number of cores to be used
        ncores = os.cpu_count()

        flat_combine = combine_images(
                flat_data_lst,
                mode        = mode,
                upper_clip  = cosmic_clip,
                maxiter     = maxiter,
                maskmode    = maskmode,
                ncores      = ncores,
                )
        info = []
        info.append(('FLAT NFILE', n_flat))
        for iframe, dataframe in enumerate(results):
            key1   = 'FLAT FILEID {:03d}'.format(iframe+1)
            value1 = dataframe.head['FILENAME']
            info.append((key1, value1))
        info.append(('FLAT COMBINE_MODE', mode))
        info.append(('FLAT COSMIC_CLIP', cosmic_clip))
        info.append(('FLAT MAXITER', maxiter))

        flat_frame = ESPADONSFrame(
                data = flat_combine,
                head = fits.Header(),
                mask = np.zeros_like(flat_combine, dtype=np.int16),
                info = info,
                )

        filename = options.get('file', None)
        if filename:
            filepath = context.midproc_path / filename
            flat_frame.save(filepath, overwrite=True)
            print('Flat saved to ', filepath)

        context[self.name] = {
                'flat': flat_frame,
                }

class TraceOrder(AnalysisPipelineStep):
    def process(self, context, inputs, **options):
        image = inputs.data

        scan_step = options.get('scan_step', 100)
        align_deg = options.get('align_deg', 2)
        fit_deg   = options.get('fit_deg', 4)

        aperset, aperset_A, aperset_B = find_apertures(
                image,
                scan_step = scan_step,
                align_deg = align_deg,
                degree    = fit_deg,
                mode      = 'normal',
                figpath   = context.figure_path,
                )

        trac_file  = context.midproc_path / 'trace.txt'
        tracA_file = context.midproc_path / 'trace_A.txt'
        tracB_file = context.midproc_path / 'trace_B.txt'

        context.aperset_path = trac_file
        context.aperset_A_path = tracA_file
        context.aperset_B_path = tracB_file

        aperset.save_txt(trac_file)
        aperset_A.save_txt(tracA_file)
        aperset_B.save_txt(tracB_file)

        self.aperset = aperset
        self.aperset_A = aperset_A
        self.aperset_B = aperset_B

        context[self.name] = {
                'trace': aperset,
                'trace_A': aperset_A,
                'trace_B': aperset_B,
                }

class GetSensMap(AnalysisPipelineStep):
    def process(self, context, inputs, **options):
        flat_frame = inputs['frame']
        aperset    = inputs['trace']

        flat_data = flat_frame.data
        flat_mask = flat_frame.mask

        fig_spatial = SpatialProfileFigure()
        sens, flatspec_lst = get_flat2(flat_data, flat_mask,
                    apertureset     = aperset,
                    nflat           = 10,
                    smooth_A_func   = smooth_aperpar_A,
                    smooth_c_func   = smooth_aperpar_c,
                    smooth_bkg_func = smooth_aperpar_bkg,
                    mode            = 'normal',
                    fig_spatial = fig_spatial,
                    )
        figname = 'spatial_profile_flat.png'
        title = 'Spatial Profile of flat'
        fig_spatial.suptitle(title)
        fig_spatial.savefig(figname)
        fig_spatial.close()

        head = fits.Header()
        sens_frame = ESPADONSFrame(data = sens,
                                   head = head,
                                   mask = flat_mask,
                                   info = flat_frame.info,
                                   )

        sens_path = context.midproc_path / 'sens.fits'
        sens_frame.save(sens_path, overwrite=True)
        context[self.name] = {
                'sens': sens_frame
                }


class CalibrateWavelength(CollectionPipelineStep):

    def finish(self, result, context, inputs, **options):
        print('Do Calibrate Wavelength')
        context[self.name] = {
                'wave': np.ones((2,2)),
                }

class ReduceScience(StreamingPipelineStep):
    def process_frame(self, frame, context):
        print('Reduce Science')



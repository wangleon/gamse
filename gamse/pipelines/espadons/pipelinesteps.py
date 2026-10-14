import os
import numpy as np
import astropy.io.fits as fits
from ...echelle.imageproc import combine_images, savitzky_golay_2d
from ..engine import (CollectionPipelineStep, AnalysisPipelineStep,
                       StreamingPipelineStep)
from ..base import ImageFrame
from .common import (select_calib_from_database,
                    BackgroundFigure, SpatialProfileFigure)
from .trace import find_apertures
from .flat import (get_flat2, smooth_aperpar_A, smooth_aperpar_c,
                   smooth_aperpar_bkg)

class ProcessBias(CollectionPipelineStep):
    def finish(self, results, context, inputs, **options):
        print('Process Bias')

        n_bias = len(results)

        bias_data_lst = np.array([result.frame.data for result in results])
        
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

        extra_head = fits.Header()
        prefix = 'HIERARCH REDUCTION BIAS '
        extra_head.append((prefix + 'NFRAMES', n_bias))
        for iframe, result in enumerate(results):
            key1   = prefix + 'FILEID {:03d}'.format(iframe+1)
            value1 = result.frame.head['FILENAME']
            extra_head.append((key1, value1))
        extra_head.append((prefix + 'COMBINE_MODE', mode))
        extra_head.append((prefix + 'COSMIC_CLIP', cosmic_clip))
        extra_head.append((prefix + 'MAXITER', maxiter))

        bias_frame = ImageFrame(
                        data = bias_combine,
                        head = fits.Header(),
                        mask = np.zeros_like(bias_combine, dtype=np.int16),
                        extra_head = extra_head,
                        )

        filename = options.get('file', None)
        if filename:
            filepath = context.midproc_path / filename
            bias_frame.save(filepath, overwrite=True)
            print('Bias saved to ', filepath)

        context.register(self.name, 'bias', bias_frame, filepath, 'image')

class ProcessFlat(CollectionPipelineStep):
    def finish(self, results, context, inputs, **options):
        print('Process Flat')

        n_flat = len(results)
        flat_data_lst = np.array([result.frame.data for result in results])

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

        extra_head = fits.Header()
        prefix = 'HIERARCH REDUCTION FLAT '
        extra_head.append((prefix + 'NFRAMES', n_flat))
        for iframe, result in enumerate(results):
            key1   = prefix + 'FILEID {:03d}'.format(iframe+1)
            value1 = result.frame.head['FILENAME']
            extra_head.append((key1, value1))
        extra_head.append((prefix + 'COMBINE_MODE', mode))
        extra_head.append((prefix + 'COSMIC_CLIP', cosmic_clip))
        extra_head.append((prefix + 'MAXITER', maxiter))

        flat_frame = ImageFrame(
                data = flat_combine,
                head = fits.Header(),
                mask = np.zeros_like(flat_combine, dtype=np.int16),
                extra_head = extra_head,
                )

        filename = options.get('file', None)
        if filename:
            filepath = context.midproc_path / filename
            flat_frame.save(filepath, overwrite=True)
            print('Flat saved to ', filepath)

        context.register(self.name, 'flat', flat_frame, filepath, 'image')

class TraceOrder(AnalysisPipelineStep):
    def process(self, context, inputs, **options):
        image = inputs.value.data

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

        aperset.save_txt(trac_file)
        aperset_A.save_txt(tracA_file)
        aperset_B.save_txt(tracB_file)

        context.register(self.name, 'trace',   aperset,   trac_file, 'aperset')
        context.register(self.name, 'trace_A', aperset_A, tracA_file, 'aperset')
        context.register(self.name, 'trace_B', aperset_B, tracB_file, 'aperset')

class GetSensMap(AnalysisPipelineStep):
    def process(self, context, inputs, **options):
        dataframe = inputs['frame'].value
        aperset   = inputs['trace'].value

        data = dataframe.data
        mask = dataframe.mask

        fig_spatial = SpatialProfileFigure()
        sens, flatspec_lst = get_flat2(data, mask,
                                       apertureset     = aperset,
                                       nflat           = 10,
                                       smooth_A_func   = smooth_aperpar_A,
                                       smooth_c_func   = smooth_aperpar_c,
                                       smooth_bkg_func = smooth_aperpar_bkg,
                                       mode            = 'normal',
                                       fig_spatial     = fig_spatial,
                                       )
        figname = 'spatial_profile_flat.png'
        title = 'Spatial Profile of flat'
        fig_spatial.suptitle(title)
        fig_spatial.savefig(figname)
        fig_spatial.close()

        head = fits.Header()
        sens_frame = ImageFrame(data = sens,
                                head = head,
                                mask = mask,
                                extra_head = dataframe.extra_head,
                                )

        sens_path = context.midproc_path / 'sens.fits'
        sens_frame.save(sens_path, overwrite=True)

        context.register(self.name, 'sens', sens_frame, sens_path, 'image')


class CalibrateWavelength(CollectionPipelineStep):

    def finish(self, results, context, inputs, **options):

        calib_lst = []
        for result in results:
            dataframe = result.frame
            calib     = result['calib']
            calib_lst.append(calib)
            fileid = dataframe.head['FILENAME']
            fname = 'wlcalib_{}.fits'.format(fileid)
            filename = context.onedspec_path / fname
            dataframe.save(filename, overwrite=True)
            print('spectrum saved to', filename)

        #context.register(self.name, 'wave', calib_lst, fname_lst, 'wave')

class ReduceScience(StreamingPipelineStep):
    def process_frame(self, frame, context):
        print('Reduce Science')

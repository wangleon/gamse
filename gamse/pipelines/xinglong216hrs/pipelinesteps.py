import os
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from ...echelle import trace
from ...echelle.imageproc import combine_images, savitzky_golay_2d
from ...echelle.trace import find_apertures
from ..engine import (CollectionPipelineStep, AnalysisPipelineStep,
                       StreamingPipelineStep)
from ..base import ImageFrame
from .common import (TraceFigure, AlignFigure, SpatialProfileFigure)
from .flat import (smooth_aperpar_A, smooth_aperpar_k, smooth_aperpar_c,
                   smooth_aperpar_bkg, get_flat)


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
        extra_head.append((prefix+'NFILE', n_bias))
        for iframe, result in enumerate(results):
            key1   = prefix + 'FILEID {:03d}'.format(iframe+1)
            value1 = result.frame.extra_head['LOGINFO FILEID']
            extra_head.append((key1, value1))
        extra_head.append((prefix+'COMBINE_MODE', mode))
        extra_head.append((prefix+'COSMIC_CLIP', cosmic_clip))
        extra_head.append((prefix+'MAXITER', maxiter))

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
        #context[self.name] = {
        #        'bias': bias_frame
        #        }

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
        extra_head.append((prefix+'NFILE', n_flat))
        for iframe, result in enumerate(results):
            key1   = prefix + 'FILEID {:03d}'.format(iframe+1)
            value1 = result.frame.extra_head['LOGINFO FILEID']
            extra_head.append((key1, value1))
        extra_head.append((prefix+'COMBINE_MODE', mode))
        extra_head.append((prefix+'COSMIC_CLIP', cosmic_clip))
        extra_head.append((prefix+'MAXITER', maxiter))

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
        #context[self.name] = {
        #        'flat': flat_frame,
        #        }

class TraceOrder(AnalysisPipelineStep):
    def process(self, context, inputs, **options):
        print('trace order')
        # inputs is a ProductRecord
        image = inputs.value.data
        mask = np.zeros_like(image, dtype=np.int16)

        scan_step  = options.get('scan_step', 100)
        minimum    = options.get('minimum', 8.0)
        separation = options.get('separation', '500:19, 1500:29, 3500:52')
        filling    = options.get('filling', 0.3)
        align_deg  = options.get('align_deg', 2)
        fit_deg    = options.get('fit_deg',   3)

        #tracefig = TraceFigure()    # create the trace figure
        #alignfig = AlignFigure()    # create the align figure
        trace_fig, trace_axes = trace.create_tracefig(
                datashape=image.shape, figsize=(12, 6), dpi=150)

        align_fig, align_axes = trace.create_alignfig(
                figsize=(12,6), dpi=150)

        aperset = find_apertures(image, mask,
                                 scan_step  = scan_step,
                                 minimum    = minimum,
                                 separation = separation,
                                 align_deg  = align_deg,
                                 filling    = filling,
                                 degree     = fit_deg,
                                 conv_core  = 10,
                                 fill       = True,
                                 fill_tol   = 10,
                                 display    = False,
                                 #fig_trace  = tracefig,
                                 #trace_axes = trace_axes,
                                 #fig_align  = alignfig,
                                 align_axes = align_axes,
                                 trace_axes = trace_axes,
                                 )
        # save the trace figure
        #tracefig.adjust_positions()
        title = 'Order Trace'
        trace_fig.suptitle(title, fontsize=15)
        trace_figname = context.figure_path / 'trace.png'
        trace_fig.savefig(trace_figname)
        trace.adjust_tracefig(trace_axes)
        plt.close(trace_fig)

        # save the alignment figure
        #align_fig.adjust_axes()
        title = 'Order Alignment'
        align_fig.suptitle(title, fontsize=12)
        align_figname = context.figure_path / 'align.png'
        align_fig.savefig(align_figname)
        plt.close(align_fig)

        aperset_filename = context.midproc_path / 'trace.trc'
        aperset_regname  = context.midproc_path / 'trace.reg'
        aperset.save_txt(aperset_filename)
        aperset.save_reg(aperset_regname)


        context.register(self.name, 'trace', aperset, aperset_filename, 'aperset')
        #context[self.name] = {'trace': aperset}

class GetSensMap(AnalysisPipelineStep):
    def process(self, context, inputs, **options):
        dataframe  = inputs['frame'].value
        aperset    = inputs['trace'].value

        data = dataframe.data
        mask = dataframe.mask

        ndisp = data.shape[1]
        p1, p2, pstep = -8, 8, 0.1
        profile_x = np.arange(p1, p2+1e-4, pstep)
        disp_x_lst = np.arange(48, ndisp, 500)
        q_threshold = 50

        fig_spatial = SpatialProfileFigure()
        sens, flatspec_lst, profile_lst = get_flat(
                data            = data,
                mask            = mask,
                apertureset     = aperset,
                nflat           = 10,
                q_threshold     = q_threshold,
                smooth_A_func   = smooth_aperpar_A,
                smooth_c_func   = smooth_aperpar_c,
                smooth_bkg_func = smooth_aperpar_bkg,
                mode            = 'debug',
                fig_spatial     = fig_spatial,
                flatname        = 'master',
                profile_x       = profile_x,
                disp_x_lst      = disp_x_lst,
                )

        figpath = context.figure_path / 'spatial_profile.png'
        title = 'Spatial Profile of Flat'
        fig_spatial.suptitle(title)
        fig_spatial.savefig(figpath)
        fig_spatial.close()

        head = fits.Header()
        sens_frame = ImageFrame(data = sens,
                                head = head,
                                mask = mask,
                                extra_head = dataframe.extra_head,
                                )

        sens_path = context.midproc_path / 'sens.fits'
        sens_frame.save(sens_path, overwrite=True)
        print('sens image saved to', sens_path)

        context.register(self.name, 'sens', sens_frame, sens_path, 'image')
        
        #context[self.name] = {
        #        'sens': sens_frame
        #        }

class CalibrateWavelength(CollectionPipelineStep):
    def finish(self, results, context, inputs, **options):

        print('Calibrate Wavelength')
        calib_lst = {}
        fname_lst = []
        for result in results:
            dataframe = result.frame
            calib     = result['calib']
            fileid    = calib['fileid']
            calib_lst[fileid] = calib
            fileid = dataframe.extra_head['LOGINFO FILEID']

            # save 1d calibrated spectrum
            fname = 'wlcalib_{}.fits'.format(fileid)
            filepath = context.midproc_path / fname
            dataframe.save(filepath, overwrite=True)

            fname_lst.append(filepath)
            print('ThAr spectrum saved to ', filepath)

        context.register(self.name, 'wave', calib_lst, fname_lst, 'wave')

class ReduceScience(StreamingPipelineStep):
    def process_frame(self, frame, context):
        print('Reduce Science')

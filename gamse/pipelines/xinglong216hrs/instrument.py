from ..base import Instrument
from ..common import BiasSubtractionStep, FlatCorrectionStep
from .pipelinesteps import *
from .framesteps import *
from .dataframe import read_dataframe

class Xinglong216HRS(Instrument):
    name = 'Xinglong216RHS'

    direction = 'xr-'  # default direction

    PIPELINE_STEPS = {
        'ProcessBias':          ProcessBias,
        'ProcessFlat':          ProcessFlat,
        'TraceOrder':           TraceOrder,
        'GetSensMap':           GetSensMap,
        'CalibrateWavelength':  CalibrateWavelength,
        'ReduceScience':        ReduceScience,
        }
    FRAME_STEPS = {
        'OverscanSubtraction':  OverscanSubtractionStep,
        'BiasSubtraction':      BiasSubtractionStep,
        'FlatCorrection':       FlatCorrectionStep,
        'ScatterSubtraction':   ScatterSubtractionStep,
        'Extraction':           ExtractionStep,
        'OptimalExtraction':    OptimalExtractionStep,
        'CalibrateWavelength':  CalibrateWavelengthStep,
        'ApplyWavelength':      ApplyWavelengthStep,
        'SaveSpectrum':         SaveSpectrumStep,
        }

    def read(self, filepath, logitem):
        return read_dataframe(filepath, logitem)


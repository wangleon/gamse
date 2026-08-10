from ..base import Instrument
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
        'Extraction':           ExtractionStep,
        'CalibrateWavelength':  CalibrateWavelengthStep,
        'ApplyWavelength':      ApplyWavelengthStep,
        }

    def read(self, filepath, logitem):
        return read_dataframe(filepath, logitem)


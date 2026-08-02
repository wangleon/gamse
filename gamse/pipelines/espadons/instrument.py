from ..base import Instrument
from .pipelinesteps import *
from .framesteps import *
from .dataframe import ESPADONSFrame

class ESPADONS(Instrument):
    name = 'ESPADONS'

    direction = 'yr+'  # default direction for ESPADONS

    PIPELINE_STEPS = {
        'ProcessBias':          ProcessBias,
        'ProcessFlat':          ProcessFlat,
        'TraceOrder':           TraceOrder,
        'GetSensMap':           GetSensMap,
        'CalibrateWavelength':   CalibrateWavelength,
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

    def read(self, filepath):
        dataframe = ESPADONSFrame.read(filepath)
        return dataframe


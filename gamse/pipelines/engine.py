import ast
import yaml
from pathlib import Path
from importlib import import_module
from abc import ABC, abstractmethod
from graphlib import TopologicalSorter

from .. import ObslogTable

def find_refs(obj):
    refs = set()
    if isinstance(obj, str):
        if obj.startswith("@"):
            refs.add(obj[1:].split(".")[0])
    elif isinstance(obj, dict):
        for value in obj.values():
            refs |= find_refs(value)
    elif isinstance(obj, list):
        for item in obj:
            refs |= find_refs(item)
    return refs


def resolve_reference(value, context=None):
    if isinstance(value, str):
        if not value.startswith("@"):
            return value
        ref = value[1:]
        parts = ref.split(".")
        step = parts[0]
        product = context[step]
        if len(parts) == 1:
            return product
        else:
            # parts are only allowed to have one value
            return product[parts[1]]
    elif isinstance(value, dict):
        return {k: resolve_reference(v, context)
                    for k, v in value.items()}
    elif isinstance(value, list):
        return [resolve_reference(v, context)
                    for v in value]

def resolve_reference_alt(value, context=None):
    if isinstance(value, str):
        if not value.startswith("@"):
            return value
        key = value[1:]
        return product[key]
    elif isinstance(value, dict):
        return {k: resolve_reference(v, context)
                    for k, v in value.items()}
    elif isinstance(value, list):
        return [resolve_reference(v, context)
                    for v in value]


class Pipeline:
    def __init__(self, instrument, filename):
        self.instrument = instrument
        with open(filename) as f:
            self.config = yaml.safe_load(f)

    def run(self, context):

        # pass instrument to data context
        context.set_instrument(self.instrument)

        pipeline_steps = self.config['pipeline']

        # resolve dependencies and generate DAG
        dag = {}
        for name, cfg in pipeline_steps.items():
            dag[name] = find_refs(cfg)
        # convert DAG to topological sorter
        ts = TopologicalSorter(dag)

        for name in ts.static_order():
            cfg = pipeline_steps[name]
            clsname = cfg['class']
            PipelineStepClass = self.instrument.PIPELINE_STEPS[clsname]
            obj = PipelineStepClass(self, name, self.config)
            #print('run pipeline:', name)
            obj.run(cfg, context)


class PipelineStep(ABC):
    def __init__(self, parent, name, config):
        self.parent = parent
        self.name   = name
        self.config = config
        self.frame_engine = FrameEngine(self, config)

    @abstractmethod
    def run(self, cfg, context):
        ...

class CollectionPipelineStep(PipelineStep):

    def run(self, cfg, context):

        selector = cfg.get('selector', {})
        newtable = context.logtable.filter(context.data_filter)
        logitems = newtable.filter(selector)

        operations = cfg.get('operations', [])

        options = cfg.get('options', {})

        if 'input' in cfg:
            inputs = resolve_reference(cfg['input'], context)
        else:
            inputs = None
        

        results = []
        for logitem in logitems:

            # get filepath to raw data
            filepath = context.find_rawdata_path(logitem)

            # read raw data 
            dataframe = self.parent.instrument.read(filepath, logitem)

            # print to console
            dataframe.print_to_console()
            

            # claim a FrameResult instance
            result = FrameResult(frame=dataframe)

            # run step
            result = self.frame_engine.run(operations, result, context)

            # append the new results
            results.append(result)

        return self.finish(results, context, inputs, **options)

    @abstractmethod
    def finish(self, results, context, inputs, **options):
        ...

class StreamingPipelineStep(PipelineStep):
    def run(self, cfg, context):

        selector = cfg.get('selector', {})
        newtable = context.logtable.filter(context.data_filter)
        logitems = newtable.filter(selector)

        operations = cfg.get('operations', [])
        if 'input' in cfg:
            inputs = resolve_reference(cfg['input'], context)
        else:
            inputs = None
        options = cfg.get('options', {})

        for logitem in logitems:

            # get filepath to raw data
            filepath = context.find_rawdata_path(logitem)

            # read raw data 
            dataframe = self.parent.instrument.read(filepath, logitem)

            # print to console
            dataframe.print_to_console()

            # claim a FrameResult instance
            result = FrameResult(frame=dataframe)

            # run step
            result = self.frame_engine.run(operations, result, context)

            #self.process_frame(dataframe, context)

        return self.finish(context)

    def finish(self, context):
        return None

    @abstractmethod
    def process_frame(self, frame, context):
        ...

class AnalysisPipelineStep(PipelineStep):
    def run(self, cfg, context):
        inputs = resolve_reference(cfg['input'], context)
        options = cfg.get('options', {})
        return self.process(context, inputs, **options)

    @abstractmethod
    def process(self, context, **options):
        ...

class FrameEngine:
    def __init__(self, parent, config):
        self.parent = parent
        self.config = config
        self.step_cfg = config['steps']
        self._class_cache = {}

    def run(self, operations, result, context):

        for op in operations:
            name = op['step']
            clsname = self.config['steps'][name]['class']

            ### find the corresponding FrameStep
            if name in self._class_cache:
                # class in cache
                FrameStepClass = self._class_cache[name]
            else:
                # not in cache
                if '.' not in name:
                    # get internal class from registrated steps in each
                    # instrument
                    instrument = self.parent.parent.instrument
                    FrameStepClass = instrument.FRAME_STEPS[clsname]
                else:
                    # external class
                    module_name, class_name = name.rsplit('.', 1)
                    module = import_module(module_name)
                    FrameStepClass = getattr(module, class_name)

                self._class_cache[name] = FrameStepClass

            # clainm this FrameStep
            step = FrameStepClass(parent=self)

            # pack options, remove 'step' option
            options = {k:v for k,v in op.items() if k != 'step'}

            #print('Run Framestep:', name, options)
            result = step.run(result, context, **options)
        return result


class FrameStep(ABC):
    def __init__(self, parent):
        self.parent = parent
        # self.parent is the FrameEngine class
        # self.parent.parent is the PipelineStep
        # self.parent.parent.parent is the Pipeline

    @abstractmethod
    def run(self, dataframe, context, **options):
        ...


class FrameResult:

    def __init__(self, frame):
        self.frame = frame
        self.outputs = {}

    def __getitem__(self, key):
        if key == 'frame':
            return self.frame
        return self.outputs[key]

    def __setitem__(self, key, value):
        if key == 'frame':
            self.frame = value
        else:
            self.outputs[key] = value

class ProductRecord:
    def __init__(self, parent, value=None, path=None, dtype=None):
        self.parent = parent
        self.value = value

        if path is None:
            self.path = None
        elif isinstance(path, str) or isinstance(path, Path):
            self.path = Path(path)
        elif isinstance(path, list):
            self.path  = [Path(_path) for _path in path]
        else:
            raise ValueError


        self.dtype = dtype

    @property
    def loaded(self):
        return self.value is not None

    def load(self):
        if self.loaded:
            return self.value

        if self.path is None:
            raise RuntimeError

        if self.dtype == 'image':
            cls = self.parent.instrument
            return cls.read_dataframe(self.path)
        else:
            print(self.parent.instrument)

        return self.value

class DataContext:

    def __init__(self, **kwargs):

        self.reset()

        rawdata_path = kwargs.pop('rawdata_path')
        self.rawdata_path = Path(rawdata_path).expanduser().resolve()

        # get reduction path
        reduction_path = kwargs.pop('reduction_path', None)
        if reduction_path is None:
            self.reduction_path = Path.cwd()
        else:
            self.reduction_path = Path(reduction_path).expanduser().resolve()

        # get midproc path
        midproc_path = kwargs.pop('midproc_path', None)
        if midproc_path is None:
            self.midproc_path = self.reduction_path / 'midproc'
        else:
            self.midproc_path = Path(midproc_path).expanduser().resolve()
        # create midproc path if not exist
        self.midproc_path.mkdir(parents=True, exist_ok=True)

        # get figure path
        figure_path = kwargs.pop('figure_path', None)
        if figure_path is None:
            self.figure_path = self.reduction_path / 'figures'
        else:
            self.figure_path = Path(figure_path).expanduser().resolve()
        # create figure path if not exist
        self.figure_path.mkdir(parents=True, exist_ok=True)

        # get onedspec path
        onedspec_path = kwargs.pop('onedspec_path', None)
        if onedspec_path is None:
            self.onedspec_path = self.reduction_path / 'onedspec'
        else:
            self.onedspec_path = Path(onedspec_path).expanduser().resolve()
        # create onedspec path if not exist
        self.onedspec_path.mkdir(parents=True, exist_ok=True)

        # read logtable
        logtablename = kwargs.pop('logtable_path')
        self.logtable_path = Path(logtablename).expanduser().resolve()
        self.logtable = ObslogTable.read(self.logtable_path,
                                   format='ascii.fixed_width_two_line')

        
        # read data filter
        data_filter = kwargs.pop('data_filter', None)
        if data_filter is None:
            self.data_filter = {}
        else:
            self.data_filter = ast.literal_eval(data_filter)

        # read rawpath patterns
        rawdata_patterns = kwargs.pop('rawdata_patterns', None)
        if rawdata_patterns is None:
            self.rawdata_patterns = []
        else:
            self.rawdata_patterns = ast.literal_eval(rawdata_patterns)

    def reset(self):
        self.instrument     = None
        self.logtable_path  = ''
        self.rawdata_path   = ''
        self.reduction_path = ''
        self.midproc_path   = ''
        self.figure_path    = ''
        self.onedspec_path  = ''
        #self.bias_path      = ''
        #self.flat_path      = ''
        #self.sens_path      = ''
        #self.aperset_path   = ''
        #self.aperset_A_path = ''
        #self.aperset_B_path = ''
        self.data_filter = {}
        self.rawdata_patterns = []
        self._products = {}

    def __contains__(self, key):
        return key in self._products

    def __getitem__(self, key):
        return self._products[key]

    def __setitem__(self, key, value):
        self._products[key] = value

    def set_filter(self, condition):
        """Set global data filter.
        """
        self.data_filter = condition

    def set_instrument(self, instrument):
        self.instrument = instrument

    def set_rawdata_patterns(self, patterns):
        self.rawdata_patterns = patterns

    def find_rawdata_path(self, logitem):
        for pattern in self.rawdata_patterns:
            fname = pattern.format(**logitem)
            filepath = self.rawdata_path / fname
            if filepath.exists():
                return filepath

    def to_dict(self):
        """Convert to dict.

        """
        result = {
                'instrument':       '' if self.instrument is None else self.instrument.name,
                'logtable_path':    str(self.logtable_path),
                'rawdata_path':     str(self.rawdata_path),
                'reduction_path':   str(self.reduction_path),
                'midproc_path':     str(self.midproc_path),
                'figure_path':      str(self.figure_path),
                'onedspec_path':    str(self.onedspec_path),
                #'verbose': self.verbose,
                #'bias_path':        str(self.bias_path),
                #'flat_path':        str(self.flat_path),
                #'sens_path':        str(self.sens_path),
                #'aperset_path':     str(self.aperset_path),
                #'aperset_A_path':   str(self.aperset_A_path),
                #'aperset_B_path':   str(self.aperset_B_path),
                'data_filter':      str(self.data_filter),
                'rawdata_patterns': str(self.rawdata_patterns),
                }

        return result

    def to_yaml(self, yaml_path):

        context_dict = self.to_dict()

        yaml_str = yaml.dump(context_dict,
                             default_flow_style = False,
                             allow_unicode      = True,
                             sort_keys          = False,
                             )

        if yaml_path is not None:
            yaml_path_obj = Path(yaml_path)
            yaml_path_obj.parent.mkdir(parents=True, exist_ok=True)
            with open(yaml_path_obj, 'w', encoding='utf-8') as f:
                f.write(yaml_str)
            return None
        else:
            return yaml_str

    @classmethod
    def from_yaml(cls, yaml_path):
        
        yaml_path_obj = Path(yaml_path)
        if not yaml_path_obj.exists():
            raise FileNotFoundError('Config file does not exist')

        with open(yaml_path_obj, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        if config_dict is None:
            config_dict = {}

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, context_dict):
        """

        Args:
            config_dict:


        Returns:
            PipelineConfig
        """
        # creat a copy of config_dict
        kwargs = context_dict.copy()

        return cls(**kwargs)

    def register(self, stepname, productname, value, path, dtype):
        product = ProductRecord(parent = self,
                                value  = value,
                                path   = path,
                                dtype  = dtype,
                                )
        if stepname not in self._products:
            self._products[stepname] = {}
        self._products[stepname][productname] = product

from model.environment import Environment
from model.data_objects import Image, Category
import yaml


class Config(Environment):
    """
    The Config class stores configuration data. It reads in a configuration file and updates the base configuration.
    """
    def __init__(self, **kwargs):
        # Load base config and config files
        self.config = yaml.load(open(kwargs.get('file', self.CONFIG_PATH)), yaml.Loader)
        self.baseConfig = yaml.load(open(self.BASE_CONFIG_PATH), yaml.Loader)

        # Set training, predict, and dataset parameters
        for key, value in self.config['Training Parameters'].items():
            self.baseConfig['Training Parameters'][key] = value
        for key, value in self.config['Predict Parameters'].items():
            self.baseConfig['Predict Parameters'][key] = value
        for key, value in self.config['Dataset Parameters'].items():
            for key2, value2 in self.config['Dataset Parameters'][key].items():
                for key3, value3 in self.config['Dataset Parameters'][key][key2].items():
                    self.baseConfig['Dataset Parameters'][key][key2][key3] = value3

        # Set other values
        for key, value in self.config.items():
            if key not in ['Training Parameters', 'Predict Parameters', 'Dataset Parameters']:
                self.baseConfig[key] = value

        self.categories = self.baseConfig['names']


class TrainingConfig(Config):
    """Sets parameters specific to training"""
    def __init__(self):
        super().__init__()
        self.project = self.baseConfig['Training Parameters']['project']
        self.name = self.baseConfig['Training Parameters']['name']
        self.epochs = self.baseConfig['Training Parameters']['epochs']
        self.model = self.baseConfig['Training Parameters']['model']
        self.time = self.baseConfig['Training Parameters']['time']
        self.patience = self.baseConfig['Training Parameters']['patience']
        self.batch = self.baseConfig['Training Parameters']['batch']
        self.imgsz = self.baseConfig['Training Parameters']['imgsz']
        self.save = self.baseConfig['Training Parameters']['save']
        self.save_period = self.baseConfig['Training Parameters']['save_period']
        self.cache = self.baseConfig['Training Parameters']['cache']
        self.device = self.baseConfig['Training Parameters']['device']
        self.workers = self.baseConfig['Training Parameters']['workers']
        self.exist_ok = self.baseConfig['Training Parameters']['exist_ok']
        self.pretrained = self.baseConfig['Training Parameters']['pretrained']
        self.optimizer = self.baseConfig['Training Parameters']['optimizer']
        self.verbose = self.baseConfig['Training Parameters']['verbose']
        self.seed = self.baseConfig['Training Parameters']['seed']
        self.deterministic = self.baseConfig['Training Parameters']['deterministic']
        self.single_cls = self.baseConfig['Training Parameters']['single_cls']
        self.rect = self.baseConfig['Training Parameters']['rect']
        self.cos_lr = self.baseConfig['Training Parameters']['cos_lr']
        self.close_mosaic = self.baseConfig['Training Parameters']['close_mosaic']
        self.resume = self.baseConfig['Training Parameters']['resume']
        self.amp = self.baseConfig['Training Parameters']['amp']
        self.fraction = self.baseConfig['Training Parameters']['fraction']
        self.profile = self.baseConfig['Training Parameters']['profile']
        self.freeze = self.baseConfig['Training Parameters']['freeze']
        self.lr0 = self.baseConfig['Training Parameters']['lr0']
        self.lrf = self.baseConfig['Training Parameters']['lrf']
        self.momentum = self.baseConfig['Training Parameters']['momentum']
        self.weight_decay = self.baseConfig['Training Parameters']['weight_decay']
        self.warmup_epochs = self.baseConfig['Training Parameters']['warmup_epochs']
        self.warmup_momentum = self.baseConfig['Training Parameters']['warmup_momentum']
        self.warmup_bias_lr = self.baseConfig['Training Parameters']['warmup_bias_lr']
        self.box = self.baseConfig['Training Parameters']['box']
        self.cls = self.baseConfig['Training Parameters']['cls']
        self.dfl = self.baseConfig['Training Parameters']['dfl']
        self.pose = self.baseConfig['Training Parameters']['pose']
        self.kobj = self.baseConfig['Training Parameters']['kobj']
        self.label_smoothing = self.baseConfig['Training Parameters']['label_smoothing']
        self.nbs = self.baseConfig['Training Parameters']['nbs']
        self.overlap_mask = self.baseConfig['Training Parameters']['overlap_mask']
        self.mask_ratio = self.baseConfig['Training Parameters']['mask_ratio']
        self.dropout = self.baseConfig['Training Parameters']['dropout']
        self.val = self.baseConfig['Training Parameters']['val']
        self.plots = self.baseConfig['Training Parameters']['plots']


class PredictConfig(Config):
    """Sets parameters specific to predicting"""
    def __init__(self):
        super().__init__()
        self.source = self.baseConfig['Predict Parameters']['source']
        self.conf = self.baseConfig['Predict Parameters']['conf']
        self.iou = self.baseConfig['Predict Parameters']['iou']
        self.imgsz = self.baseConfig['Predict Parameters']['imgsz']
        self.half = self.baseConfig['Predict Parameters']['half']
        self.device = self.baseConfig['Predict Parameters']['device']
        self.max_det = self.baseConfig['Predict Parameters']['max_det']
        self.vid_stride = self.baseConfig['Predict Parameters']['vid_stride']
        self.stream_buffer = self.baseConfig['Predict Parameters']['stream_buffer']
        self.visualize = self.baseConfig['Predict Parameters']['visualize']
        self.augment = self.baseConfig['Predict Parameters']['augment']
        self.agnostic_nms = self.baseConfig['Predict Parameters']['agnostic_nms']
        self.classes = self.baseConfig['Predict Parameters']['classes']
        self.retina_masks = self.baseConfig['Predict Parameters']['retina_masks']
        self.embed = self.baseConfig['Predict Parameters']['embed']

        self.show = self.baseConfig['Predict Parameters']['show']
        self.save = self.baseConfig['Predict Parameters']['save']
        self.save_frames = self.baseConfig['Predict Parameters']['save_frames']
        self.save_txt = self.baseConfig['Predict Parameters']['save_txt']
        self.save_conf = self.baseConfig['Predict Parameters']['save_conf']
        self.save_crop = self.baseConfig['Predict Parameters']['save_crop']
        self.show_labels = self.baseConfig['Predict Parameters']['show_labels']
        self.show_conf = self.baseConfig['Predict Parameters']['show_conf']
        self.show_boxes = self.baseConfig['Predict Parameters']['show_boxes']
        self.line_width = self.baseConfig['Predict Parameters']['line_width']


class DatasetConfig(Config):
    """
    Sets parameters specific to the dataset. This includes what classes or images to keep or drop during the initial
    reading or merging stages
    """

    def __init__(self):
        super().__init__()
        # Get initial and merge data
        self.LoadInitialAnnotations = self.baseConfig['Dataset Parameters']['LoadInitialAnnotations']
        self.MergeAutoAnnotations = self.baseConfig['Dataset Parameters']['MergeAutoAnnotations']

        data_obj_modifications = [self.LoadInitialAnnotations, self.MergeAutoAnnotations]
        # Loop through stages
        for stage in data_obj_modifications:
            # Loop through the possible data objects
            for obj in ['Images', 'Categories']:
                # Check if data object in configuration
                if obj in stage:
                    # Get list of modifications to data object
                    mods = list(stage[obj].keys())
                    # Loop through list of possible modifications
                    for mod in ['keep', 'drop']:
                        # Check if mod in configuration
                        if mod in mods:
                            kd = stage[obj][mod]
                            # Process image and category keep/drop input
                            if type(kd) is str and kd != 'all':
                                kd_processed = []
                                groups = kd.split(', ')
                                for group in groups:
                                    num_range = [int(x) for x in group.split('-')]
                                    kd_processed.extend(list(range(num_range[0], num_range[1] + 1)))
                                kd = kd_processed

                            stage[obj][mod] = kd

    def update(self, annotationObj, stage: str) -> [[Image], [Image], [Category], [Category]]:
        """Updates Image and Category objects in an Annotation object depending on the dataset configuration"""

        lia, maa = self.LoadInitialAnnotations, self.MergeAutoAnnotations
        for mod in [lia['Images'], maa['Images']]:
            if mod['keep'] == 'all':
                mod['keep'] = [x.id for x in annotationObj.images]
        for mod in [lia['Categories'], maa['Categories']]:
            if mod['keep'] == 'all':
                mod['keep'] = [x.name for x in annotationObj.categories]

        if stage == 'train':
            drop_images = lia['Images']['drop']
            keep_images = lia['Images']['keep']

            drop_categories = lia['Categories']['drop']
            keep_categories = lia['Categories']['keep']
        else:
            drop_images = maa['Images']['drop']
            keep_images = maa['Images']['keep']

            drop_categories = maa['Categories']['drop']
            keep_categories = maa['Categories']['keep']

        return keep_images, drop_images, keep_categories, drop_categories


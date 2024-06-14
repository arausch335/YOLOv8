import os


class Environment(object):
    """Defines the path variables for the rest of the classes"""
    ROOT_DIR = os.getcwd()

    CONFIGS_DIR = os.path.join(ROOT_DIR, 'configs')
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    MODEL_DIR = os.path.join(ROOT_DIR, 'model')
    RUNS_DIR = os.path.join(ROOT_DIR, 'runs')
    WEIGHTS_DIR = os.path.join(ROOT_DIR, 'weights')

    VIDEOS_DIR = os.path.join(DATA_DIR, 'videos')
    IMAGES_DIR = os.path.join(DATA_DIR, 'images')
    COCO_ANNOTATIONS_DIR = os.path.join(DATA_DIR, 'original_annotations')
    YOLO_DATASET_DIR = os.path.join(DATA_DIR, 'yolo_dataset')
    YOLO_ANNOTATIONS_DIR = os.path.join(DATA_DIR, 'yolo_annotations')
    AUTO_ANNOTATIONS_DIR = os.path.join(DATA_DIR, 'auto_annotations')

    COCO_ANNOTATIONS_PATH = os.path.join(COCO_ANNOTATIONS_DIR, 'instances_default.json')
    CONFIG_PATH = os.path.join(CONFIGS_DIR, 'config.yaml')
    BASE_CONFIG_PATH = os.path.join(CONFIGS_DIR, 'baseConfig.yaml')

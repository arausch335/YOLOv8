from model.dataset import Dataset
from model.config import TrainingConfig, PredictConfig, DatasetConfig
from model.environment import Environment
from ultralytics import YOLO
from ultralytics.data.annotator import auto_annotate
import os
import shutil


class Model(Environment):
    """
    The Model class contains the main functions necessary to train a YOLO model and use it for predictions.
    """
    def __init__(self):
        # Sets configurations for training and predicting modes
        self.training_config = TrainingConfig()
        self.predict_config = PredictConfig()

        # Gets project name
        self.name = self.training_config.name

        # Prepares dataset for training
        self.dataset_config = DatasetConfig()
        self.dataset = Dataset(self.dataset_config)
        self.dataset.prepare()

    def train(self, **kwargs):
        """Train the model on the processed dataset using custom parameters"""
        name = kwargs.get('name', self.name)

        # Train model
        model = YOLO(os.path.join(self.WEIGHTS_DIR, 'yolov8m-seg.pt'))
        results = model.train(
            data=self.CONFIG_PATH,
            project=self.training_config.project,
            name=name,
            epochs=self.training_config.epochs,
            model=self.training_config.model,
            time=self.training_config.time,
            patience=self.training_config.patience,
            batch=self.training_config.batch,
            imgsz=self.training_config.imgsz,
            save=self.training_config.save,
            save_period=self.training_config.save_period,
            cache=self.training_config.cache,
            device=self.training_config.device,
            workers=self.training_config.workers,
            exist_ok=self.training_config.exist_ok,
            pretrained=self.training_config.pretrained,
            optimizer=self.training_config.optimizer,
            verbose=self.training_config.verbose,
            seed=self.training_config.seed,
            deterministic=self.training_config.deterministic,
            single_cls=self.training_config.single_cls,
            rect=self.training_config.rect,
            cos_lr=self.training_config.cos_lr,
            close_mosaic=self.training_config.close_mosaic,
            resume=self.training_config.resume,
            amp=self.training_config.amp,
            fraction=self.training_config.fraction,
            profile=self.training_config.profile,
            freeze=self.training_config.freeze,
            lr0=self.training_config.lr0,
            lrf=self.training_config.lrf,
            momentum=self.training_config.momentum,
            weight_decay=self.training_config.weight_decay,
            warmup_epochs=self.training_config.warmup_epochs,
            warmup_momentum=self.training_config.warmup_momentum,
            warmup_bias_lr=self.training_config.warmup_bias_lr,
            box=self.training_config.box,
            cls=self.training_config.cls,
            dfl=self.training_config.dfl,
            pose=self.training_config.pose,
            kobj=self.training_config.kobj,
            label_smoothing=self.training_config.label_smoothing,
            nbs=self.training_config.nbs,
            overlap_mask=self.training_config.overlap_mask,
            mask_ratio=self.training_config.mask_ratio,
            dropout=self.training_config.dropout,
            val=self.training_config.val,
            plots=self.training_config.plots
        )

    def predict(self, **kwargs):
        """Predict instances using pre-trained YOLO model"""
        name = kwargs.get('name', self.name)

        # Get images to run predictions on
        imageDir = kwargs.get('imageDir', self.IMAGES_DIR)
        images = os.listdir(imageDir)

        # Get model type
        modelType = kwargs.get('modelType', 'pt')
        if modelType == 'tensor_rt':
            modelPath = 'model.engine'
        else:
            modelPath = 'best.pt'

        # Predict instances
        model = YOLO(os.path.join(self.RUNS_DIR, rf'train\{name}\weights\{modelPath}'))
        results = model(source=[os.path.join(imageDir, x) for x in images],
                        conf=self.predict_config.conf,
                        iou=self.predict_config.iou,
                        imgsz=self.predict_config.imgsz,
                        half=self.predict_config.half,
                        device=self.predict_config.device,
                        max_det=self.predict_config.max_det,
                        vid_stride=self.predict_config.vid_stride,
                        stream_buffer=self.predict_config.stream_buffer,
                        visualize=self.predict_config.visualize,
                        augment=self.predict_config.augment,
                        agnostic_nms=self.predict_config.agnostic_nms,
                        classes=self.predict_config.classes,
                        retina_masks=self.predict_config.retina_masks,
                        embed=self.predict_config.embed,
                        show=self.predict_config.show,
                        save=self.predict_config.save,
                        save_frames=self.predict_config.save_frames,
                        save_txt=self.predict_config.save_txt,
                        save_conf=self.predict_config.save_conf,
                        save_crop=self.predict_config.save_crop,
                        show_labels=self.predict_config.show_labels,
                        show_conf=self.predict_config.show_conf,
                        show_boxes=self.predict_config.show_boxes,
                        line_width=self.predict_config.line_width)

        # Move predictions under runs\predict\name
        predict_path = os.path.join(self.RUNS_DIR, r'predict')
        if not os.path.exists(predict_path):
            os.mkdir(predict_path)
        os.rename(os.path.join(self.RUNS_DIR, r'segment\predict'), os.path.join(predict_path, name))
        shutil.rmtree(os.path.join(self.RUNS_DIR, r'segment'))

    def auto_annotate(self, **kwargs):
        """Predict instances of each class and create labels containing mask coordinates"""

        # Get name and images
        name = kwargs.get('name', self.name)
        imageDir = kwargs.get('imageDir', self.IMAGES_DIR)

        # Run auto annotation function if auto annotations do not already exist
        annotate_dir = os.path.join(self.RUNS_DIR, rf'annotate')
        if not os.path.exists(annotate_dir):
            os.mkdir(annotate_dir)
        if not os.path.exists(os.path.join(annotate_dir, name)):
            auto_annotate(data=imageDir,
                          det_model=os.path.join(self.RUNS_DIR, rf'train\{name}\weights\best.pt'),
                          sam_model=os.path.join(self.WEIGHTS_DIR, 'sam_b.pt'),
                          output_dir=os.path.join(annotate_dir, name),
                          device=0)
        else:
            print('Auto annotations already created')

        # Process auto annotations
        self.dataset.process_auto_annotation_results(name)

    def evaluate(self):
        pass

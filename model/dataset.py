from model.environment import Environment
from model.annotation import Annotation
from model.config import Config, DatasetConfig
import cv2
import shutil
import os


class Dataset(Environment):
    """
    A Dataset class prepares the various data types for training by the Model object. It contains multiple Annotation
    objects and their corresponding images and categories.
    """

    def __init__(self, datasetConfig: Config):
        # Define data lists
        self.videos = []
        self.images = []
        self.categories = []

        # Define annotations
        self.original_annotations = Annotation()
        self.auto_annotations = Annotation()
        self.merged_annotations = Annotation()

        # Set the configuration
        self.config = datasetConfig

    def prepare(self, overwrite=False):
        """Converts the data into formats usable by the YOLO model"""

        # Get frames from video if necessary
        if len(os.listdir(self.VIDEOS_DIR)) > 0:
            # Make the image directory
            if not os.path.exists(self.IMAGES_DIR):
                os.mkdir(self.IMAGES_DIR)
            # Get frames
            if len(os.listdir(self.IMAGES_DIR)) == 0 or overwrite is True:
                self.clean(images=True)
                self.getFrames(self.VIDEOS_DIR, self.IMAGES_DIR)

        # Update original annotations object
        self.original_annotations.load_from_COCO_json(config=self.config)

        # Convert COCO format to YOLO format and split dateset if necessary
        if not os.path.exists(self.YOLO_DATASET_DIR) or overwrite:
            self.clean(yolo_dataset=True)
            self.original_annotations.split_annotations()

    def process_auto_annotation_results(self, name: str):
        """Converts the results of the auto annotation function into a format usable by CVAT.
        Also simplifies the annotations so they contain fewer points and are easier to work with. """

        # Load annotations into Annotation object from a YOLO labels directory. Must add a reference annotation to get
        # image and class data, as they are not available in the YOLO label format.
        self.auto_annotations.batch_load_from_YOLO_dir(os.path.join(self.RUNS_DIR, rf"annotate\{name}"),
                                                       reference_annotation=self.original_annotations)

        # Merge the original annotations with the auto annotations, preserving the original annotations
        self.merged_annotations = self.merge_annotations(self.original_annotations, self.auto_annotations,
                                                         config=self.config)

        # TODO: Add functionality to snap annotations together
        # self.merged_annotations.snap_segments('Pulmonary Artery', 'Aorta')

        # Simplify annotations by removing points along same line
        self.merged_annotations.simplify_segments(tolerance=1.5)

        # Save auto annotations to JSON if they do not already exist
        if not os.path.exists(self.AUTO_ANNOTATIONS_DIR):
            os.mkdir(self.AUTO_ANNOTATIONS_DIR)
            self.merged_annotations.export_to_COCO_json(os.path.join(
                self.AUTO_ANNOTATIONS_DIR, 'simplified_merged_annotations.json'))

    def clean(self, yolo_dataset=True, images=False, videos=False, original_annotations=False,
              auto_annotations=True):
        """Removes old data"""

        if yolo_dataset and os.path.exists(self.YOLO_DATASET_DIR):
            shutil.rmtree(self.YOLO_DATASET_DIR)

        if auto_annotations and os.path.exists(self.AUTO_ANNOTATIONS_DIR):
            shutil.rmtree(self.AUTO_ANNOTATIONS_DIR)

        if images and os.path.exists(self.IMAGES_DIR):
            shutil.rmtree(self.IMAGES_DIR)
            os.mkdir(self.IMAGES_DIR)

        if videos:
            shutil.rmtree(self.VIDEOS_DIR)
            os.mkdir(self.VIDEOS_DIR)

        if original_annotations:
            shutil.rmtree(self.COCO_ANNOTATIONS_DIR)
            os.mkdir(self.COCO_ANNOTATIONS_DIR)

    @staticmethod
    def getFrames(videoDirPath: str, dstDirPath: str):
        """Extract frames (as .PNG) from video"""

        for video in os.listdir(videoDirPath):
            # Convert to open-cv video object
            vidcap = cv2.VideoCapture(f'{videoDirPath}/{video}')
            success, image = vidcap.read()

            # Save the frame as a .PNG with the CVAT image naming system
            count = 0
            while success:
                cv2.imwrite(f'{dstDirPath}/frame_{str(count).zfill(6)}.PNG', image)
                success, image = vidcap.read()
                print('Read a new frame: ', success)
                count += 1

    @staticmethod
    def merge_annotations(base_annotation: Annotation, new_annotation: Annotation, **kwargs) -> Annotation:
        """Merge two Annotation objects, reusing class and image data from the base annotation.
        Annotations from base annotation will not be overwritten by new annotation"""

        # Get which classes of the new annotation to merge from the config file
        config = kwargs.get('config', DatasetConfig())
        _, _, keep, drop = config.update(base_annotation, 'merge')

        transferred_categories = []
        for category in base_annotation.categories:
            if category.name in keep and category.name not in drop:
                transferred_categories.append(category.name)

        # Add unique annotations to list of base annotations
        merged_annotations = base_annotation.annotations.copy()
        for annotation in new_annotation.annotations:
            if (base_annotation.get_annotation(image_id=annotation.image_id, category_id=annotation.category_id) is None
                    and base_annotation.get_category(category_id=annotation.category_id).name in transferred_categories):
                merged_annotations.append(annotation)

        # Make new annotation object with new annotations, and images and categories from base annotation
        merged_annotation = Annotation(files=list({*base_annotation.files, *new_annotation.files}),
                                       images=base_annotation.images,
                                       categories=base_annotation.categories,
                                       annotations=merged_annotations)

        return merged_annotation




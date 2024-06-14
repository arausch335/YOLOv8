from model.data_objects import *
from model.environment import Environment
from model.config import DatasetConfig
from ultralytics.utils.ops import segments2boxes
from shapely.ops import snap
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import random
import json
import shutil
import os
import numpy as np
import math


class Annotation(Environment):
    """
    The Annotation class handles the annotations on a set of images. These annotations can be in either the COCO JSON
    format or the YOLO txt format.

    An Annotation object consists of a list of files, images, categories, and annotations.
    """

    def __init__(self, **kwargs):
        # Initialize an empty Annotation object
        self.files = kwargs.get('files', [])
        self.images = kwargs.get('images', [])
        self.categories = kwargs.get('categories', [])
        self.annotations = kwargs.get('annotations', [])
        self.labels = kwargs.get('labels', [])

    ####################################################################################
    # Load or export annotations
    ####################################################################################
    def load_from_COCO_json(self, **kwargs):
        """Create Annotation object containing a list of Categories, Images, and SegmentAnnotations from a
        COCO JSON file"""

        # Get and load annotations file
        annotationFile = kwargs.get('annotationFile', self.COCO_ANNOTATIONS_PATH)
        annotation = json.load(open(annotationFile))
        self.files.append(annotationFile)

        # Add categories
        for category in annotation['categories']:
            catObj = Category(name=category['name'],
                              category_id=category['id'])
            self.categories.append(catObj)

        # Add images
        for image in annotation['images']:
            imageObj = Image(file_name=image['file_name'],
                             image_id=image['id'],
                             width=image['width'],
                             height=image['height'])
            self.images.append(imageObj)

        # Get which classes of the JSON annotation to load from the config file
        config = kwargs.get('config', DatasetConfig())
        keep_images, drop_images, keep_categories, drop_categories = config.update(self, 'train')

        # Add annotations
        for a in annotation['annotations']:
            # Edit attributes dictionary, or create one if it does not exist
            try:
                a['attributes']['format'] = 'COCO'
            except KeyError:
                a['attributes'] = {'format': 'COCO'}

            # Get the category name from its id
            category_name = self.get_category(category_id=a['category_id']).name

            # Create the SegmentAnnotation object using the various data
            aObj = SegmentAnnotation(annotation_id=a['id'],
                                     image_id=a['image_id'],
                                     category_id=a['category_id'],
                                     bbox=a['bbox'],
                                     area=a['area'],
                                     segmentation=a['segmentation'][0],
                                     attributes=a['attributes'])

            # Add to annotations if specified in config
            if (a['image_id'] in keep_images and a['image_id'] not in drop_images) and (
                    category_name in keep_categories and category_name not in drop_categories):
                self.annotations.append(aObj)

    def export_to_COCO_json(self, fileName):
        """Exports Annotation objects into the COCO JSON format"""

        # Create data dictionary
        data = {
            'categories': [{'id': c.id, 'name': c.name, 'supercategory': c.supercategory}
                           for c in self.categories],
            'images': [{'id': i.id, 'file_name': i.file_name, 'width': i.width, 'height': i.height}
                       for i in self.images],
            'annotations': [{'id': a.annotation_id, 'image_id': a.image_id, 'category_id': a.category_id,
                             'area': a.area, 'bbox': list(a.bbox), 'segmentation': [a.segmentation],
                             'iscrowd': 0, 'attributes': a.attributes}
                            for a in self.annotations]
        }

        # Write to file
        with open(fileName, 'w') as f:
            json.dump(data, f)

    def load_from_YOLO_label(self, labelFile: str, **kwargs):
        """Create Annotation object containing a list of Categories, Images, and SegmentAnnotations from a
        YOLO label file. NOTE: image and category data are usually not present, so a reference Annotation object or a
        list of class and image objects must be passed in"""

        # Get reference categories and images
        categories = kwargs.get('categories', [])
        images = kwargs.get('images', [])
        if categories == [] and images == []:
            reference_annotation = kwargs.get('reference_annotation', Annotation())
            categories = reference_annotation.categories
            images = reference_annotation.images

        # Add image
        image_name = f"{os.path.basename(labelFile).split('.')[0]}.PNG"
        image = self.get_image(file_name=image_name, image_list=images)
        self.images.append(image)

        # Add and read file
        self.files.append(labelFile)
        label = open(labelFile).readlines()

        # Add annotation
        for annotation in label:
            # Add category
            category_id = int(annotation.split(' ')[0]) + 1
            category = self.get_category(category_id=category_id, category_list=categories)
            if category not in self.categories:
                self.categories.append(category)

            # Create segment from normalized x and y coordinates
            xn, yn = annotation.split(' ')[1::2], annotation.split(' ')[2::2]
            x = [round(float(a) * image.width, 2) for a in xn]
            y = [round(float(a) * image.height, 2) for a in yn]
            segment = get_polygon_coordinates(x, y)

            # Calculate the bounding box of the segment
            seg = np.array([list(a) for a in list(zip(x, y))])
            seg.reshape(-1, 2)
            bbox = segments2boxes([seg])[0]

            # Calculate the area of the segment
            poly = make_polygon_from_segment(segment)
            area = poly.area

            # Get the annotation id
            a = (image.id * random.randint(50, 100)) ** category.id
            b = math.log10(a)
            c = a / (category.id * b)
            d = math.log10(c / (image.id ** category.id))
            e = round(d % 1, 8) * (10 ** 8)
            annotation_id = int(e)

            # Create SegmentAnnotation object using various data
            aObj = SegmentAnnotation(annotation_id=annotation_id,
                                     image_id=image.id,
                                     category_id=category.id,
                                     bbox=bbox,
                                     area=area,
                                     segmentation=segment,
                                     attributes={'format': 'YOLO'})

            # Add annotation
            self.annotations.append(aObj)

    def batch_load_from_YOLO_dir(self, labelDir: str, **kwargs):
        """Add YOLO annotations to Annotation object from a directory containing multiple label files"""

        # Get reference data
        categories = kwargs.get('categories', [])
        images = kwargs.get('images', [])
        reference_annotation = kwargs.get('reference_annotation', Annotation())

        # Get label files in directory
        labels = os.listdir(labelDir)
        for label in labels:
            # Load annotation
            self.load_from_YOLO_label(os.path.join(labelDir, label), categories=categories, images=images,
                                      reference_annotation=reference_annotation)

    def export_to_YOLO_label(self, **kwargs):
        """Exports Annotation objects into multiple YOLO label formats"""

        # Get directories
        outputDir = kwargs.get('outputDir', self.YOLO_ANNOTATIONS_DIR)
        IMAGES_DIR = os.path.join(outputDir, 'images')
        LABELS_DIR = os.path.join(outputDir, 'labels')

        # Make directories if they do not exist
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)
        if not os.path.exists(IMAGES_DIR):
            os.mkdir(IMAGES_DIR)
            os.mkdir(LABELS_DIR)

        # Get images as each YOLO file is organized by image
        images = kwargs.get('images', self.images)
        for image in images:
            # Get annotations corresponding to image
            annotations = self.get_annotations(image_id=image.id)

            # Skip unannotated images
            if len(annotations) == 0:
                continue

            # Set variable for file contents
            yoloOutput = []
            for annotation in annotations:
                # Convert segment into normalized x and y stream
                x, y = annotation.x, annotation.y
                xn, yn = [a / image.width for a in x], [b / image.height for b in y]
                poly = get_polygon_coordinates(xn, yn)

                # Combine category id with segment to form one line of the file
                line = [annotation.category_id - 1, *[round(x, 6) for x in poly]]
                yoloOutput.append(' '.join([str(x) for x in line]))

            # Create the file contents
            yoloString = '\n'.join(yoloOutput)

            # Write to the file
            f = open(os.path.join(LABELS_DIR, f"{image.file_name.split('.')[0]}.txt"), 'w')
            f.write(yoloString)
            f.close()

            # Copy the corresponding image into the directory
            shutil.copy(os.path.join(self.IMAGES_DIR, image.file_name), IMAGES_DIR)

    ####################################################################################
    # Retrieve object using identifier
    ####################################################################################
    def get_image(self, image_id=None, file_name=None, **kwargs) -> Image:
        """Get image from id or file name"""

        image_list = kwargs.get('image_list', self.images)
        for image in image_list:
            if image.id == image_id:
                return image
            elif image.file_name == file_name:
                return image
        return None

    def get_category(self, category_id=None, category_name=None, **kwargs) -> Category:
        """Get category from id or category name"""

        category_list = kwargs.get('category_list', self.categories)
        for category in category_list:
            if category.id == category_id:
                return category
            elif category.name == category_name:
                return category
        return None

    def get_annotation(self, image_id=None, category_id=None, image_name=None, category_name=None) \
            -> SegmentAnnotation:
        """Get single annotation from image id and category id"""

        # Convert names into ids
        if image_name is not None:
            image_id = self.get_image(file_name=image_name).id
        if category_name is not None:
            category_id = self.get_category(category_name=category_name).id

        for annotation in self.annotations:
            if annotation.image_id == image_id and annotation.category_id == category_id:
                return annotation
        return None

    def get_annotations(self, image_id=None, category_id=None, image_name=None, category_name=None) \
            -> [SegmentAnnotation]:
        """Get all annotations from image id or category id"""

        # Convert names into ids
        if image_name is not None:
            image_id = self.get_image(file_name=image_name).id
        if category_name is not None:
            category_id = self.get_category(category_name=category_name).id

        annotations = []
        for annotation in self.annotations:
            if annotation.image_id == image_id:
                annotations.append(annotation)
            if annotation.category_id == category_id:
                annotations.append(annotation)

        return annotations

    ####################################################################################
    # Modify annotations
    ####################################################################################
    def split_annotations(self, split=(0.85, 0.15, 0.0)):
        """Split images and annotations into train, val, and test directories"""

        # Set directory paths
        TRAIN_DIR = os.path.join(self.YOLO_DATASET_DIR, 'train')
        VAL_DIR = os.path.join(self.YOLO_DATASET_DIR, 'val')
        TEST_DIR = os.path.join(self.YOLO_DATASET_DIR, 'test')

        # Create directories if they do not exist
        if not os.path.exists(self.YOLO_DATASET_DIR):
            os.makedirs(TRAIN_DIR)
            os.makedirs(VAL_DIR)
            os.makedirs(TEST_DIR)

        # Get images
        image_list = []
        for image in self.images.copy():
            if len(self.get_annotations(image_name=image.file_name)) > 0:
                image_list.append(image)

        # Get training images
        train_images = random.sample(image_list, int(split[0] * len(image_list)))
        self.export_to_YOLO_label(outputDir=TRAIN_DIR, images=train_images)

        # Get validation images
        new_ratio = round(split[1] / (1 - split[0]), 2)
        for x in train_images.copy(): image_list.remove(x)
        val_images = random.sample(image_list, int(new_ratio * len(image_list)))
        self.export_to_YOLO_label(outputDir=VAL_DIR, images=val_images)

        # Get testing images
        new_ratio = split[-1] / (1 - split[0] - split[1])
        for x in val_images: image_list.remove(x)
        test_images = random.sample(image_list, int(new_ratio * len(image_list)))
        self.export_to_YOLO_label(outputDir=TEST_DIR, images=test_images)

    def drop(self, image_ids=None, category_ids=None, image_names=None, category_names=None):
        """Drop annotations from certain images or containing certain frames"""

        dropped_annotations = []
        
        if image_ids is not None:
            for image_id in image_ids:
                dropped_annotations.extend(self.get_annotations(image_id=image_id))
        
        if category_ids is not None:
            for category_id in category_ids:
                dropped_annotations.extend(self.get_annotations(category_id=category_id))
        
        if image_names is not None:
            for image_name in image_names:
                dropped_annotations.extend(self.get_annotations(image_name=image_name))
        
        if category_names is not None:
            for category_name in category_names:
                dropped_annotations.extend(self.get_annotations(category_name=category_name))
    
        for annotation in dropped_annotations:
            self.annotations.remove(annotation)

    def simplify_segments(self, tolerance: float):
        """Simplify segments by removing similar points"""
        for annotation in self.annotations:
            annotation.segmentation = simplify_polygon(annotation.segmentation, tolerance)

    def snap_segments(self, base_category, edit_category):
        """Snap segments to one another if they are bordering"""
        # TODO: Not currently working
        base_category_ID = self.get_category(category_name=base_category).id
        edit_category_ID = self.get_category(category_name=edit_category).id

        for image in self.images:
            image_id = image.id
            base_annotation = self.get_annotation(image_id=image_id, category_id=base_category_ID)
            edit_annotation = self.get_annotation(image_id=image_id, category_id=edit_category_ID)

            base_poly = make_polygon_from_segment(base_annotation.segmentation)
            edit_poly = make_polygon_from_segment(edit_annotation.segmentation)

            edit_annotation.segmentation = make_segment_from_polygon(snap(base_poly, edit_poly, 0.5))

    ####################################################################################
    # Area calculations
    ####################################################################################
    def get_mask_areas(self, categories: [Category]) -> [[float]]:
        """Returns the areas of the masks per specified category"""

        areas = []
        for category in categories:
            category_areas = []
            for annotation in self.annotations:
                category_name = self.get_category(category_id=annotation.category_id).name
                if category_name == category:
                    category_areas.append(annotation.area)
            areas.append(category_areas)
        return areas

    def get_bbox_wh_ratio(self, categories: [Category]) -> [[float]]:
        """Returns the ratios of the width and height of the bounding boxes per specified category"""
        ratios = []
        for category in categories:
            category_ratios = []
            for annotation in self.annotations:
                category_name = self.get_category(category_id=annotation.category_id).name
                if category_name == category:
                    bbox = annotation.bbox
                    wh = bbox[-2] / bbox[-1]
                    category_ratios.append(wh)
            ratios.append(category_ratios)
        return ratios

    def get_frames_at_peaks(self, data: [float], plot=False, makeFolder=False):
        """Saves the images where data is at a local maximum"""

        # Get peaks in data and corresponding image ids
        peaks, _ = find_peaks(data, distance=17)
        image_ids = [x + 1 for x in peaks]

        # Get Image objects
        peak_images = []
        for image in self.images:
            if image.id in image_ids:
                peak_images.append(image)

        # Plot peaks if specified
        if plot:
            plt.plot(data)
            plt.plot(peaks, [data[peak] for peak in peaks], "x")
            plt.show()

        # Save images to folder if specified
        if makeFolder:
            peak_path = os.path.join(self.DATA_DIR, 'peak_images')
            os.mkdir(peak_path)
            for image in peak_images:
                image_path = os.path.join(self.IMAGES_DIR, image.file_name)
                shutil.copy(image_path, peak_path)

    ####################################################################################
    # Display
    ####################################################################################
    def display(self):
        """Print information about Annotation object"""
        print(f'Images: {len(self.images)}')
        print(f'\t{", ".join([str(x.id) for x in self.images])}')
        print(f'\nCategories: {len(self.categories)}')
        print(f'\t{", ".join([f"{x.name} ({x.id})" for x in self.categories])}')

        annotation = random.choice(self.annotations)
        print(f'\nRandom Annotation: {annotation.annotation_id}')
        print(f'\tImage: {self.get_image(annotation.image_id).file_name} ({annotation.image_id})')
        print(f'\tCategory: {self.get_category(annotation.category_id).name} ({annotation.category_id})')
        print(f'\tSegment Length: {len(annotation.segmentation)}')
        print(f'\tBounding Box: {", ".join([str(x) for x in list(annotation.bbox)])}')
        print(f'\tArea: {annotation.area}')

    def query(self, image_id=None, category_id=None, image_name=None, category_name=None):
        """Display information about annotations at a certain image or of a certain category"""
        if image_name is not None:
            image_id = self.get_image(file_name=image_name).id
        if category_name is not None:
            category_id = self.get_category(category_name=category_name).id

        annotations = []
        for annotation in self.annotations:
            if annotation.image_id == image_id:
                annotations.append(annotation)
            if annotation.category_id == category_id:
                annotations.append(annotation)

        for annotation in annotations:
            print(f'\nAnnotation: {annotation.annotation_id}')
            print(f'\tImage: {self.get_image(annotation.image_id).file_name} ({annotation.image_id})')
            print(f'\tCategory: {self.get_category(annotation.category_id).name} ({annotation.category_id})')
            print(f'\tSegment Length: {len(annotation.segmentation)}')
            print(f'\tBounding Box: {", ".join([str(x) for x in list(annotation.bbox)])}')
            print(f'\tArea: {annotation.area}')

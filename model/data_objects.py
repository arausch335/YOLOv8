import shapely


class Category(object):
    def __init__(self, name, category_id, supercategory=''):
        self.name = name
        self.id = category_id
        self.supercategory = supercategory


class Image(object):
    def __init__(self, file_name, image_id, width, height):
        self.file_name = file_name
        self.id = image_id
        self.width = width
        self.height = height


class SegmentAnnotation(object):
    def __init__(self, annotation_id, image_id, category_id, bbox, area, segmentation, attributes):
        self.annotation_id = annotation_id
        self.image_id = image_id
        self.category_id = category_id
        self.bbox = bbox
        self.area = area
        self.segmentation = segmentation
        self.attributes = attributes

        self.x, self.y = get_xy_coordinates(self.segmentation)


def get_xy_coordinates(segment):
    return segment[::2], segment[1::2]


def get_polygon_coordinates(x, y):
    return sum(map(list, zip(x, y)), [])


def make_polygon_from_segment(segment):
    x, y = get_xy_coordinates(segment)
    polygon = shapely.Polygon(list(zip(x, y)))
    return polygon


def make_segment_from_polygon(polygon):
    xy = polygon.exterior.xy
    x, y = list(xy[0]), list(xy[1])
    return get_polygon_coordinates(x, y)


def simplify_polygon(segment, tolerance):
    polygon = make_polygon_from_segment(segment)
    simplified_poly = polygon.simplify(tolerance)
    simplified_seg = make_segment_from_polygon(simplified_poly)
    # print(len(segment), len(simplified_seg))
    return simplified_seg

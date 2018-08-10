import os

import cv2


def list_dir_recursive(d):
    return [os.path.join(dp, f) for dp, dn, fn in os.walk(d) for f in fn]


class DataHandler:
    def __init__(self, images_dir='images'):
        self.images_dir = images_dir
        file_paths = list_dir_recursive(self.images_dir)
        image_paths = [p for p in file_paths if '.tif' in p]
        self.gt_path = list(set(file_paths) - set(image_paths))[0]
        reference_images = [p for p in image_paths if 'reference' in p]
        inspected_images = [p.replace('reference', 'inspected') for p in reference_images]
        assert set(inspected_images).issubset(set(image_paths))
        self.image_pairs = list(zip(reference_images, inspected_images))

    def get(self):
        for image_pair in self.image_pairs:
            yield [cv2.imread(p) for p in image_pair]


class DefectDetector:
    def __init__(self):
        pass


def main():
    data_handler = DataHandler()
    for image_pair in data_handler.get():
        print(image_pair)


if __name__ == '__main__':
    main()

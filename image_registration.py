import numpy as np
import scipy


class TranslationTransform:
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy

    def transform(self, image):
        return scipy.ndimage.interpolation.shift(image, [self.dy, self.dx])

    def get_valid_mask(self, image_shape):
        dx, dy = self.dx, self.dy
        mask = np.ones(image_shape, dtype=np.bool)
        if dy > 0:
            dy = np.int(np.ceil(dy))
            mask[:dy, :] = False
        else:
            dy = np.int(np.floor(dy))
            mask[dy:, :] = False
        if dx > 0:
            dx = np.int(np.ceil(dx))
            mask[:, :dx] = False
        else:
            dx = np.int(np.floor(dx))
            mask[:, dx:] = False
        return mask

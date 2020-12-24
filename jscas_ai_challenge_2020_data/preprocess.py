from enum import Enum, auto
import cv2
import numpy as np

class ColorSpace(Enum):
    BGR = auto()
    RGB = auto()
    HSV = auto()
    GRAY = auto()

class ImagePreprocessor:
    def __init__(self, original_images, color=ColorSpace.BGR):
        # original images have bgr color space
        # original_images.shape is (image_num, height, width, channel)
        self.original_images = original_images
        self.color = ColorSpace.BGR
        if original_images.shape[3] != 3: # if gray
            self.color = ColorSpace.GRAY
    # size.shape is (width, height)
    def preprocess(self, size, color=None, do_normalize=True):
        images = self.__resize(self.original_images, size)
        
        if self.color == color:
            print(f'Already color space is {color}.')
        elif self.color == ColorSpace.GRAY: # if gray
            print('image color is gray')
        elif color is None:
            pass
        else:
            images = self.__cnvcolor(images, color)
        
        if do_normalize:
            images = self.__normalize(images)
        return images
        
    def __resize(self, images, size):
        return np.asarray([cv2.resize(x, size, interpolation=cv2.INTER_AREA) for x in images], dtype=np.float32)
    
    def __cnvcolor(self, images, color):
        convert_enum = None
        
        if self.color == ColorSpace.BGR:
            if color == ColorSpace.RGB:
                convert_enum = cv2.COLOR_BGR2RGB
            elif color == ColorSpace.HSV:
                convert_enum = cv2.COLOR_BGR2HSV
        elif self.color == ColorSpace.RGB:
            if color == ColorSpace.GBR:
                convert_enum = cv2.COLOR_RGB2GBR
            elif color == ColorSpace.HSV:
                convert_enum = cv2.COLOR_RGB2HSV
        return np.array([cv2.cvtColor(x, convert_enum) for x in images])

    def __normalize(self, images):
        normalized_images = np.empty_like(images)
        for i in range(images.shape[3]):
            max_val = np.max(images[:, :, :, i])
            normalized_images[:, :, :, i] = images[:, :, :, i] / max_val
        return normalized_images
        

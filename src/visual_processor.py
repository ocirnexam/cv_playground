import cv2 as cv
import numpy as np

class ImageProcessor():
    """
    Class for processing and manipulating images using opencv2
    """

    def __init__(self, file):
        self.__initial_file = self.__read_image(file)
        self.__file         = np.copy(self.__initial_file)
        self.__width        = self.__file.shape[1]
        self.__height       = self.__file.shape[0]

    
    def __read_image(self, file):
        """
        reads an image and returns pixel matrix
        """
        return cv.imread(file)
    

    def __avg_pixel_value(self, y, x):
        return (int(self.__file[y,x,0]) + int(self.__file[y,x,1]) + int(self.__file[y,x,2])) // 3
    

    def threshold(self, value=100):
        """
        turns image pixels to black if average of pixel is not reaching threshold
        """
        for y in range(self.__height):
            for x in range(self.__width):
                self.__file[y,x] = self.__file[y,x] if self.__avg_pixel_value(y, x) > value else [0, 0, 0]
    

    def threshold_bw(self, value=100):
        """
        turns image into black and white with threshold as border
        """
        for y in range(self.__height):
            for x in range(self.__width):
                self.__file[y,x] = [255, 255, 255] if self.__avg_pixel_value(y, x) > value else [0, 0, 0]


    def grayscale(self):
        """
        Converts image to grayscale
        """
        self.__file = cv.cvtColor(self.__file, cv.COLOR_BGR2GRAY)

    
    def reset(self):
        """
        Reset pixel matrix to initial
        """
        self.__file   = np.copy(self.__initial_file)
        self.__width  = self.__file.shape[1]
        self.__height = self.__file.shape[0]

    
    def rescale(self, scale=0.75):
        """
        rescale image to specific size
        """
        assert(scale <= 1.0 and scale >= 0)
        self.__width   = int(self.__file.shape[1] * scale)
        self.__height  = int(self.__file.shape[0] * scale)
        
        self.__file    = cv.resize(self.__file, (self.__width, self.__height), interpolation=cv.INTER_AREA)


    def show(self):
        """
        prints a pixel matrix as image on screen
        """
        cv.imshow("image", self.__file)
        cv.waitKey(0)
        cv.destroyAllWindows()


class VideoProcessor():
    """
    Class for processing and manipulating videos using opencv2
    """

    def __init__(self, file, scale=1.0):
        self.__file = self.__read_video(file)
        if scale < 1.0:
            self.rescale(scale)


    def __read_video(self, file):
        """
        reads a video and returns array of pixel matrices
        """
        capture     = cv.VideoCapture(file)
        frame_count = capture.get(7)
        frames      = []
        for i in range(int(frame_count)):
            isTrue, frame = capture.read()
            if isTrue:
                frames.append(frame)
        capture.release()
        return frames
    

    def rescale(self, scale=0.75):
        """
        rescale video to specific size
        """
        assert(scale <= 1.0 and scale >= 0)
        width      = int(self.__file[0].shape[1] * scale)
        height     = int(self.__file[0].shape[0] * scale)
        dimensions = (width, height)
        for i in range(len(self.__file)):
            self.__file[i] = cv.resize(self.__file[i], dimensions, interpolation=cv.INTER_AREA)


    def show(self):
        """
        prints an array of pixel matrices as video on screen
        """
        for v in self.__file:
            cv.imshow("Video", v)
            cv.waitKey(20)
        cv.destroyAllWindows()
    
#!/usr/bin/env python3
import numpy as np
import os
import cv2
from image_geometry import PinholeCameraModel

class Augmenter:

    # the following is partially taken from the dt-core/packages/image_processing/include/image_processing package

    def __init__(self, im_width, im_height, homography, camera_info):
        self.im_width = im_width
        self.im_height = im_height
        self.H = np.array(homography).reshape((3, 3))
        self.Hinv = np.linalg.inv(self.H)
        self.ci = camera_info
        self.pcm = PinholeCameraModel()
        self.pcm.fromCameraInfo(self.ci)
        self._rectify_inited = False
        self._distort_inited = False

    def process_image(self, cv_image_raw, interpolation=cv2.INTER_NEAREST):
        ''' Undistort an image.
            To be more precise, pass interpolation= cv2.INTER_CUBIC
        '''
        if not self._rectify_inited:
            self._init_rectify_maps()
        #
        #        inter = cv2.INTER_NEAREST  # 30 ms
        #         inter = cv2.INTER_CUBIC # 80 ms
        #         cv_image_rectified = np.zeros(np.shape(cv_image_raw))
        cv_image_rectified = np.empty_like(cv_image_raw)
        res = cv2.remap(cv_image_raw, self.mapx, self.mapy, interpolation,
                        cv_image_rectified)
        return res

    def _init_rectify_maps(self):
        W = self.pcm.width
        H = self.pcm.height
        mapx = np.ndarray(shape=(H, W, 1), dtype='float32')
        mapy = np.ndarray(shape=(H, W, 1), dtype='float32')
        mapx, mapy = cv2.initUndistortRectifyMap(self.pcm.K, self.pcm.D, self.pcm.R,
                                                 self.pcm.P, (W, H),
                                                 cv2.CV_32FC1, mapx, mapy)
        self.mapx = mapx
        self.mapy = mapy
        self._rectify_inited = True

    def ground2pixel(self, point):
        """
        Projects a point on the ground plane to a normalized pixel (``[0, 1] X [0, 1]``) using the homography matrix.
        Args:
            point (:py:class:`Point`): A :py:class:`Point` object on the ground plane. Only the ``x`` and ``y`` values are used.
        Returns:
            :py:class:`Point` : A :py:class:`Point` object in normalized coordinates. Only the ``x`` and ``y`` values are used.
        Raises:
            ValueError: If the input point's ``z`` attribute is non-zero. The point must be on the ground (``z=0``).
        """
        # if point.z != 0:
        #     msg = 'This method assumes that the point is a ground point (z=0). '
        #     msg += 'However, the point is (%s,%s,%s)' % (point.x, point.y, point.z)
        #     raise ValueError(msg)

        ground_point = np.array([point[0], point[1], 1.0])
        image_point = np.dot(self.Hinv, ground_point)
        image_point = image_point / image_point[2]

        pixel = Point()
        pixel.x = int(image_point[0])
        pixel.y = int(image_point[1])

        return pixel
        
    def render_segments(self, image, file):
        """ renders lines into an image as specified by a yaml file,
        via a loop over all specified desired segments. 
        """

        # read point "names" from the yaml file
        for name in file["points"]:
            try:
                points.update({name : file["points"][name][1]})
            except:
                points = {name : file["points"][name][1]}
            
        # iterate over all segments and draw them onto the image
        for i in range(len(file["segments"])):
            x,y = file["segments"][i]["points"]
            x,y = [points[x], points[y]]
            x_pixel = self.ground2pixel(x)
            y_pixel = self.ground2pixel(y)
            color = file["segments"][i]["color"]
            rend_img = self.draw_segment(image, x_pixel, y_pixel, color)
            
        return rend_img

    def draw_segment(self, image, pt_x, pt_y, color):
        defined_colors = {
            'red': ['rgb', [1, 0, 0]],
            'green': ['rgb', [0, 1, 0]],
            'blue': ['rgb', [0, 0, 1]],
            'yellow': ['rgb', [1, 1, 0]],
            'magenta': ['rgb', [1, 0 , 1]],
            'cyan': ['rgb', [0, 1, 1]],
            'white': ['rgb', [1, 1, 1]],
            'black': ['rgb', [0, 0, 0]]}
        _color_type, [r, g, b] = defined_colors[color]
        cv2.line(image, (pt_x.x, pt_x.y), (pt_y.x, pt_y.y), (b * 255, g * 255, r * 255), 5)
        return image

class Point:
    """
    Point class. Convenience class for storing ROS-independent 3D points.
    """
    def __init__(self, x=None, y=None, z=None):
        self.x = x  #: x-coordinate
        self.y = y  #: y-coordinate
        self.z = z  #: z-coordinate

    @staticmethod
    def from_message(msg):
        """
        Generates a class instance from a ROS message. Expects that the message has attributes ``x`` and ``y``.
        If the message additionally has a ``z`` attribute, it will take it as well. Otherwise ``z`` will be set to 0.
        Args:
            msg: A ROS message or another object with ``x`` and ``y`` attributes
        Returns:
            :py:class:`Point` : A Point object
        """
        x = msg.x
        y = msg.y
        try:
            z = msg.z
        except AttributeError:
            z = 0
        return Point(x, y, z)
    


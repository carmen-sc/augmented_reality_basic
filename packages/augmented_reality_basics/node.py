#!/usr/bin/env python3
import numpy as np
import os
import copy
import rospy
import cv2
from cv_bridge import CvBridge
import yaml

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from sensor_msgs.msg import CameraInfo, CompressedImage
from sensor_msgs.srv import SetCameraInfo, SetCameraInfoResponse

from augmented_reality_basics_module import Augmenter

class AugmentedRealityNode(DTROS):

    def __init__(self, node_name):
        """Augmented Reality Node
        Draws segments on an image given a map via a specified yaml file.
        """

        # Initialize the DTROS parent class
        super(AugmentedRealityNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        self.veh_name = rospy.get_namespace().strip("/")
        self.node_name = node_name
        self.map_file = rospy.get_param("~map_path")
        self.map_name = rospy.get_param("~map_name").split('.')[0]
        self.map = self.readYamlFile(self.map_file)
        
                
        # grab values of ros parameters
        self._res_w = 640
        self._res_h = 480

        # Load calibration files
        self.calibration_call()
        
        # Subscribe to the compressed camera image
        self.cam_image = rospy.Subscriber(f'/{self.veh_name}/camera_node/image/compressed', CompressedImage, self.callback, queue_size=10)
        
        # Publishers
        self.aug_image = rospy.Publisher(f'/{self.veh_name}/{self.node_name}/{self.map_name}/image/compressed', CompressedImage, queue_size=10)
        
        self.bridge = CvBridge()
        self.augmenter = Augmenter(self._res_w, self._res_h, self.homography, self.current_camera_info)

        self.log("Initialized")
        

    def callback(self, image):
        """ Callback method that writes the augmented image to the appropriate topic.
        """

        image = self.bridge.compressed_imgmsg_to_cv2(image)

        #undistort the image
        image = self.augmenter.process_image(image)
        
        # render the segments onto the image
        image = self.augmenter.render_segments(image, self.map)
        
        #publish image to new topic
        self.aug_image.publish(self.bridge.cv2_to_compressed_imgmsg(image))


    def readYamlFile(self,fname):
        """
        Reads the YAML file in the path specified by 'fname'.
        E.G. :
            the calibration file is located in : `/data/config/calibrations/filename/DUCKIEBOT_NAME.yaml`
        """
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file)
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                        %(fname, exc), type='fatal')
                rospy.signal_shutdown()
                return

    def calibration_call(self):
        # copied from the camera driver (dt-duckiebot-interface)

        # For intrinsic calibration
        self.cali_file_folder = '/data/config/calibrations/camera_intrinsic/'
        self.frame_id = rospy.get_namespace().strip('/') + '/camera_optical_frame'
        self.cali_file = self.cali_file_folder + rospy.get_namespace().strip("/") + ".yaml"

        # Locate calibration yaml file or use the default otherwise
        if not os.path.isfile(self.cali_file):
            self.logwarn("Calibration not found: %s.\n Using default instead." % self.cali_file)
            self.cali_file = (self.cali_file_folder + "default.yaml")

        # Shutdown if no calibration file not found
        if not os.path.isfile(self.cali_file):
            rospy.signal_shutdown("Found no calibration file ... aborting")

        # Load the calibration file
        self.original_camera_info = self.load_camera_info(self.cali_file)
        self.original_camera_info.header.frame_id = self.frame_id
        self.current_camera_info = copy.deepcopy(self.original_camera_info)
        self.update_camera_params()
        self.log("Using calibration file: %s" % self.cali_file)


        # extrinsic calibration
        self.ex_cali_file_folder = '/data/config/calibrations/camera_extrinsic/'
        self.ex_cali_file = self.ex_cali_file_folder + rospy.get_namespace().strip("/") + ".yaml"
        self.ex_cali = self.readYamlFile(self.ex_cali_file)
        
        # define homography
        self.homography = self.ex_cali["homography"]

    def update_camera_params(self):
        """ Update the camera parameters based on the current resolution.
        The camera matrix, rectification matrix, and projection matrix depend on
        the resolution of the image.
        As the calibration has been done at a specific resolution, these matrices need
        to be adjusted if a different resolution is being used.
        TODO: Test that this really works.
        """

        scale_width = float(self._res_w) / self.original_camera_info.width
        scale_height = float(self._res_h) / self.original_camera_info.height

        scale_matrix = np.ones(9)
        scale_matrix[0] *= scale_width
        scale_matrix[2] *= scale_width
        scale_matrix[4] *= scale_height
        scale_matrix[5] *= scale_height

        # Adjust the camera matrix resolution
        self.current_camera_info.height = self._res_h
        self.current_camera_info.width = self._res_w

        # Adjust the K matrix
        self.current_camera_info.K = np.array(self.original_camera_info.K) * scale_matrix

        # Adjust the P matrix (done by Rohit)
        scale_matrix = np.ones(12)
        scale_matrix[0] *= scale_width
        scale_matrix[2] *= scale_width
        scale_matrix[5] *= scale_height
        scale_matrix[6] *= scale_height
        self.current_camera_info.P = np.array(self.original_camera_info.P) * scale_matrix

    @staticmethod
    def load_camera_info(filename):
        """Loads the camera calibration files.
        Loads the intrinsic camera calibration.
        Args:
            filename (:obj:`str`): filename of calibration files.
        Returns:
            :obj:`CameraInfo`: a CameraInfo message object
        """
        with open(filename, 'r') as stream:
            calib_data = yaml.load(stream)
        cam_info = CameraInfo()
        cam_info.width = calib_data['image_width']
        cam_info.height = calib_data['image_height']
        cam_info.K = calib_data['camera_matrix']['data']
        cam_info.D = calib_data['distortion_coefficients']['data']
        cam_info.R = calib_data['rectification_matrix']['data']
        cam_info.P = calib_data['projection_matrix']['data']
        cam_info.distortion_model = calib_data['distortion_model']
        return cam_info
        


if __name__ == '__main__':
    node = AugmentedRealityNode(node_name='augmented_reality_basics_node')
    # Keep it spinning to keep the node alive
    rospy.spin()
    rospy.loginfo("augmented_reality_basics_node is up and running...")
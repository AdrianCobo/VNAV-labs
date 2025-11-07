#!/usr/bin/env python3

import numpy as np
import copy

import rclpy
from rclpy.node import Node

import tf2_ros

from std_msgs.msg import Header, String
from sensor_msgs.msg import Image as ImageMsg
from sensor_msgs.msg import Imu, CameraInfo, LaserScan
from nav_msgs.msg import Odometry
from mav_msgs.msg import Actuators
from geometry_msgs.msg import (
    Pose,
    PoseStamped,
    Point,
    PointStamped,
    TransformStamped,
    Quaternion,
    Twist,
    PoseWithCovarianceStamped,
    Vector3Stamped,
)
from ackermann_msgs.msg import AckermannDriveStamped
from rosgraph_msgs.msg import Clock
from cv_bridge import CvBridge, CvBridgeError

import tesse_ros_bridge.utils
from tesse_ros_bridge.noise_simulator import NoiseParams, NoiseSimulator

from tesse_msgs.srv import (
    SceneRequestService,
    ObjectSpawnRequestService,
    RepositionRequestService,
)
from tesse_msgs.msg import CollisionStats

from tesse_ros_bridge.consts import *

from tesse.msgs import *
from tesse.env import *
from tesse.utils import *


class TesseQuadrotorControlInterface(Node):
    def __init__(self):
        """This class provides a ROS 2 interface for controlling TESSE quadrotor agents."""
        super().__init__('TesseQuadrotorControlInterface_node')

        # Networking parameters
        self.sim_ip = self.declare_parameter("sim_ip", "127.0.0.1").value
        self.self_ip = self.declare_parameter("self_ip", "127.0.0.1").value
        self.use_broadcast = self.declare_parameter("use_broadcast", False).value
        self.position_port = self.declare_parameter("position_port", 9000).value
        self.metadata_port = self.declare_parameter("metadata_port", 9001).value
        self.image_port = self.declare_parameter("image_port", 9002).value
        self.udp_port = self.declare_parameter("udp_port", 9004).value
        self.step_port = self.declare_parameter("step_port", 9005).value
        self.scan_port = self.declare_parameter("lidar_port", 9006).value
        self.scan_udp_port = self.declare_parameter("lidar_udp_port", 9007).value

        # Topics
        self.props_speeds_topic = "rotor_speed_cmds"

        # Initialize the Env object to communicate with simulator
        self.env = Env(
            simulation_ip=self.sim_ip,
            own_ip=self.self_ip,
            position_port=self.position_port,
            metadata_port=self.metadata_port,
            image_port=self.image_port,
            step_port=self.step_port,
        )

        # setup control interface subscriber
        self.props_speeds_sub = self.create_subscription(
            Actuators,
            self.props_speeds_topic,
            self.props_control_cb,
            10  # QoS depth
        )

    def props_control_cb(self, msg: Actuators):
        """Callback function used for propeller speed control

        :param msg: A mav_msgs.msg.Actuators message. The field angular_velocities is used for setting the propeller speeds.
        """
        speeds = msg.angular_velocities
        self.env.send(PropSpeeds(speeds[0], speeds[1], speeds[2], speeds[3]))
        #self.get_logger().info(f"Propeller speeds: {speeds}")


def main(args=None):
    rclpy.init(args=args)
    node = TesseQuadrotorControlInterface()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

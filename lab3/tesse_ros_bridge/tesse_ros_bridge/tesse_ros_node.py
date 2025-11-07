#!/usr/bin/env python3

import numpy as np
import copy

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

# import tf
import tf2_ros
from std_msgs.msg import Header, String
from sensor_msgs.msg import Image as ImageMsg
from sensor_msgs.msg import Imu, CameraInfo, LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PoseStamped, Point, \
     PointStamped, TransformStamped, Quaternion, \
     Twist, PoseWithCovarianceStamped, Vector3Stamped
from ackermann_msgs.msg import AckermannDriveStamped
from rosgraph_msgs.msg import Clock
from cv_bridge import CvBridge, CvBridgeError

import tesse_ros_bridge.utils
from tesse_ros_bridge.noise_simulator import NoiseParams, NoiseSimulator

from tesse_msgs.srv import SceneRequestService, \
     ObjectSpawnRequestService, RepositionRequestService
from tesse_msgs.msg import CollisionStats
from tesse_ros_bridge.consts import *

from tesse.msgs import *
from tesse.env import *
from tesse.utils import *

import tf_transformations

from tf2_ros.buffer import Buffer
from rclpy.duration import Duration
import builtin_interfaces.msg

def from_sec_float(seconds):
    secs = int(seconds)
    nsecs = int((seconds - secs) * 1e9)
    return builtin_interfaces.msg.Time(sec = secs, nanosec = nsecs)

def mk_transform(trans, quat, timestamp, child_frame_id, frame_id):
    t = TransformStamped()
    t.header.stamp = timestamp
    t.header.frame_id = frame_id
    t.child_frame_id = child_frame_id

    t.transform.translation.x = trans[0]
    t.transform.translation.y = trans[1]
    t.transform.translation.z = trans[2]

    t.transform.rotation.x = quat[0]
    t.transform.rotation.y = quat[1]
    t.transform.rotation.z = quat[2]
    t.transform.rotation.w = quat[3]

    return t

class TesseROSWrapper(Node):

    def __init__(self):
        super().__init__('tesse_ros_bridge')

        qos10 = QoSProfile(depth=10)

        # Interface parameters
        self.step_mode_enabled = self.declare_parameter("enable_step_mode", False).value

        # Networking parameters
        self.sim_ip        = self.declare_parameter("sim_ip", "127.0.0.1").value
        self.self_ip       = self.declare_parameter("self_ip", "127.0.0.1").value
        self.use_broadcast = self.declare_parameter("use_broadcast", False).value
        self.position_port = self.declare_parameter("position_port", 9000).value
        self.metadata_port = self.declare_parameter("metadata_port", 9001).value
        self.image_port    = self.declare_parameter("image_port", 9002).value
        self.udp_port      = self.declare_parameter("udp_port", 9004).value
        self.step_port     = self.declare_parameter("step_port", 9005).value
        self.scan_port     = self.declare_parameter("lidar_port", 9006).value
        self.scan_udp_port = self.declare_parameter("lidar_udp_port", 9007).value

        # Set data to publish
        self.publish_clock             = self.declare_parameter("publish_clock", False).value
        self.publish_metadata          = self.declare_parameter("publish_metadata", False).value
        self.publish_collisions        = self.declare_parameter("publish_collisions", False).value
        self.publish_imu               = self.declare_parameter("publish_imu", False).value
        self.publish_odom              = self.declare_parameter("publish_odom", False).value
        self.publish_noisy_imu         = self.declare_parameter("publish_noisy_imu", False).value
        self.publish_imu_noise_biases  = self.declare_parameter("publish_imu_noise_biases", False).value
        self.publish_noisy_odom        = self.declare_parameter("publish_noisy_odom", False).value
        self.publish_stereo_rgb        = self.declare_parameter("publish_stereo_rgb", False).value
        self.publish_stereo_gry        = self.declare_parameter("publish_stereo_gry", False).value
        self.publish_segmentation      = self.declare_parameter("publish_segmentation", False).value
        self.publish_depth             = self.declare_parameter("publish_depth", False).value
        self.publish_third_pov         = self.declare_parameter("publish_third_pov", False).value
        self.publish_front_lidar       = self.declare_parameter("publish_front_lidar", False).value
        self.publish_rear_lidar        = self.declare_parameter("publish_rear_lidar", False).value

        # Simulator speed parameters
        self.frame_rate     = self.declare_parameter("frame_rate", 20.0).value
        self.imu_rate       = self.declare_parameter("imu_rate", 200.0).value
        self.scan_rate      = self.declare_parameter("scan_rate", 200.0).value

        # Output parameters
        self.use_gt_frames        = self.declare_parameter("use_gt_frames", False).value
        self.world_frame_id       = self.declare_parameter("world_frame_id", "world").value
        self.body_frame_id        = self.declare_parameter("body_frame_id", "base_link").value
        self.body_frame_id_gt     = self.declare_parameter("body_frame_id_gt", "base_link_gt").value
        self.map_frame_id         = self.declare_parameter("map_frame_id", "map").value

        # Init noisification parameters
        try:
            self.noise_params = NoiseParams(self)
        except TypeError:
            self.noise_params = NoiseParams()

        # Init noise simulator
        self.noise_simulator = NoiseSimulator(self.noise_params)

        self.imu_gyro_bias_vec  = Vector3Stamped()
        self.imu_accel_bias_vec = Vector3Stamped()
        if self.use_gt_frames:
            self.imu_gyro_bias_vec.header.frame_id  = self.body_frame_id_gt
            self.imu_accel_bias_vec.header.frame_id = self.body_frame_id_gt
        else:
            self.imu_gyro_bias_vec.header.frame_id  = self.body_frame_id
            self.imu_accel_bias_vec.header.frame_id = self.body_frame_id

        # Initialize the Env object to communicate with simulator
        self.env = Env(simulation_ip=self.sim_ip,
                       own_ip=self.self_ip,
                       position_port=self.position_port,
                       metadata_port=self.metadata_port,
                       image_port=self.image_port,
                       step_port=self.step_port)

        # Setup ROS services
        self.setup_ros_services()

        # Setup simulator step mode with teleop
        if self.step_mode_enabled:
            self.last_step_cmd = []
            self.env.send(SetFrameRate(self.frame_rate))

        # Setup collision
        enable_collision = self.declare_parameter("enable_collision", True).value
        if not enable_collision:
            self.setup_collision(enable_collision)

        # Change scene
        initial_scene = self.declare_parameter("initial_scene", 1).value
        ####################################
        # Disabled while migrating to ROS2 #
        ####################################
        # rospy.wait_for_service('scene_change_request')
        # self.change_scene(initial_scene)

        # TODO(marcus): this is not nice! Need a return from the sim!
        # Suddenly the problem went away. Keeping for when it comes back...
        # import time
        # time.sleep(5)

        # To send images via ROS network and convert from/to ROS
        self.cv_bridge = CvBridge()

        # Transform broadcasters
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.tf_buffer = Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Don't call static_tf_broadcaster.sendTransform multiple times.
        # Rather call it once with multiple static tfs! Check issue #40
        self.static_tfs_to_broadcast = []
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        # Setup all sensor data publishers and sensor objects for interfacing
        self.cameras = []
        self.img_pubs = []
        self.cam_info_pubs = []
        self.cam_info_msgs = []
        self.cam_params = []
        self.lidars = []
        self.scan_pubs = []
        self.lidar_params = []

        self.setup_all_cameras()
        self.setup_all_lidars()

        # Set up static transform between map and world
        # TODO(marcus): would be good to have this for all scene environments!
        # self.setup_static_map_tf()

        # Publish all sensor static TFs
        if self.static_tfs_to_broadcast:
            self.static_tf_broadcaster.sendTransform(self.static_tfs_to_broadcast)

        # Setup metadata publisher
        if self.publish_metadata:
            self.metadata_pub = self.create_publisher(String, "metadata", qos_profile=qos10)

        # Setup lidar UdpListener
        # TODO(marcus): see if this can be done as a UDP high-rate broadacst
        # TODO(marcus): add optional `use_broadcast` arg to this one too
        # self.scan_listener = UdpListener(port=self.scan_udp_port,
        #                                  rate=self.scan_rate)
        # self.scan_listener.subscribe('scan_subscriber', self.scan_cb)

        # If the clock updates faster than images can be queried in
        # step mode, the image callback is called twice on the same
        # timestamp which leads to duplicate published images.
        # Track image timestamps to prevent this
        self.last_image_timestamp = None

        # Setup ROS publishers for metadata
        if self.publish_imu:
            self.clean_imu_pub = self.create_publisher(Imu, "imu/clean/imu", qos_profile=qos10)

        if self.publish_odom:
            self.odom_pub = self.create_publisher(Odometry, "odom", qos_profile=qos10)

        if self.publish_noisy_imu:
            self.noisy_imu_pub      = self.create_publisher(Imu, "imu/noisy/imu", qos_profile=qos10)
            self.imu_gyro_bias_pub  = self.create_publisher(Vector3Stamped, "imu/noisy/biases/gyro", qos_profile=qos10)
            self.imu_accel_bias_pub = self.create_publisher(Vector3Stamped, "imu/noisy/biases/accel", qos_profile=qos10)

        if self.publish_noisy_odom:
            self.noisy_odom_pub = self.create_publisher(Odometry, "odom/noisy", qos_profile=qos10)

        if self.publish_collisions:
            self.coll_pub = self.create_publisher(CollisionStats, "collision", qos_profile=qos10)

        # Required states for finite difference calculations
        self.prev_time      = 0.0
        self.prev_vel_brh   = [0.0, 0.0, 0.0]
        self.prev_enu_R_brh = np.identity(3)

        # Spawn initial objects
        self.spawn_initial_objects()

        # Setup metadata UdpListener
        udp_host = self.self_ip
        if self.use_broadcast:
            udp_host = '<broadcast>'
        self.meta_listener = UdpListener(host=udp_host,
                                         port=self.udp_port,
                                         rate=self.imu_rate)
        self.meta_listener.subscribe('udp_subscriber', self.meta_cb)

        # Simulated time requires that we constantly publish to '/clock'.
        if self.publish_clock:
            from rosgraph_msgs.msg import Clock as RosClock
            self.clock_pub = self.create_publisher(RosClock, "/clock", qos_profile=qos10)

        # Setup initial-pose subscriber
        self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.reposition_cb, 10)

        # Reposition agent if needed
        # time.sleep(2)  # TODO(marcus): only necessary for play_traj! not sure why?
        """
        # TODO @fishberg
        # set in tesse_agent_params.yaml -- currently ['enabled'] = False, so I'm just deleting this for now
        init_pose_dict = rospy.get_param("~initial_pose", {})
        if init_pose_dict['enabled']:
            print("TESSE_ROS_NODE: Repositioning to custom initial pose.", )
            init_pose = Pose()
            init_pose.position.x = init_pose_dict['px']
            init_pose.position.y = init_pose_dict['py']
            init_pose.position.z = init_pose_dict['pz']
            init_pose.orientation.x = init_pose_dict['qx']
            init_pose.orientation.y = init_pose_dict['qy']
            init_pose.orientation.z = init_pose_dict['qz']
            init_pose.orientation.w = init_pose_dict['qw']
            self.reposition_agent(init_pose)
        """

        # Setup driving commands
        ackermann_drive = self.declare_parameter("drive_with_ackermann", False).value

        if ackermann_drive:
            self.create_subscription(AckermannDriveStamped, "drive", self.cmd_cb_ackermann, 10)
        else:
            self.create_subscription(Twist, "drive", self.cmd_cb_twist, 10)

        self.get_logger().info("TESSE_ROS_NODE: Initialization complete.")

    def spin(self):
        """ Start timers and callbacks.
            Because we are publishing sim time, we
            cannot simply call `rclpy.spin()` as this will wait for messages
            to go to /clock first, and will freeze the node.
        """
        self.meta_listener.start()

        if self.get_parameter("num_objects").value > 0:
            self.create_timer(1.0 / self.frame_rate, self.object_cb)

        if len(self.cameras) > 0:
            self.create_timer(1.0 / self.frame_rate, self.image_cb)

        if len(self.lidars) > 0:
            self.create_timer(1.0 / self.scan_rate, self.scan_cb_slow)

        self.create_timer(1.0 / self.frame_rate, self.clock_cb)

    def clock_cb(self, event):
        """ Publishes simulated clock time as well as collision statistics.

            Gets current metadata from the simulator over the low-rate metadata
            port. Publishes the timestamp. Collision messages are also published
            here because the metadata broadcast does not check for collisions.

            Args:
                event: a dummy parameter (kept for compatibility)
        """
        if self.step_mode_enabled:
            if len(self.last_step_cmd) > 0:
                cur_cmd = self.last_step_cmd.pop(0)
                self.env.send(StepWithForce(force_z=cur_cmd[0],
                                            torque_y=cur_cmd[2],
                                            force_x=cur_cmd[1]))
            else:
                self.get_logger().info("TESSE_ROS_NODE: No commands to publish...")

        if self.publish_clock or self.publish_collisions:
            try:
                sim_data = self.env.request(MetadataRequest()).metadata
                metadata = tesse_ros_bridge.utils.parse_metadata(sim_data)

                if self.publish_clock:
                    curr_ros_time = from_sec_float(metadata['time'])
                    c = Clock()
                    c.clock = curr_ros_time
                    self.clock_pub.publish(c)

                # Publish collision statistics if necessary
                if self.publish_collisions and metadata['collision_status']:
                    coll_msg = CollisionStats()
                    coll_msg.header.frame_id = self.body_frame_id_gt
                    coll_msg.header.stamp = curr_ros_time
                    coll_msg.is_collision = metadata['collision_status']
                    coll_msg.object_name = metadata['collision_object']
                    self.coll_pub.publish(coll_msg)

            except Exception as error:
                self.get_logger().error(f"TESSE_ROS_NODE: clock_cb error: {error}")

    def cmd_cb_ackermann(self, msg):
        """ Listens to published drive commands and sends to simulator.
            Works with the included key_teleop script.

            Sets simulator speed, turn speed, and then publishes a drive
            and turn flag based on message contents. Note that these commands
            are translated into forces and torques on the car. Note this only
            works with tesse agents that use car-like dynamics.

            Args:
                msg: An ackermann_msgs/AckermannDriveStamped message.
        """
        turn_angle_deg = np.rad2deg(msg.drive.steering_angle)
        speed = msg.drive.speed

        # Messages are in SI, RCC requires KMH:
        speed_kmh = speed * 3.6

        self.env.send(SetSpeed(np.abs(speed_kmh)))
        self.env.send(SetTurnSpeed(turn_angle_deg))

        # Move forward/backwards
        if speed_kmh > 0:
            self.env.send(Drive(1))
        elif speed_kmh < 0:
            self.env.send(Drive(-1))

        # Turn left/right
        if turn_angle_deg > 0:
            self.env.send(Turn(1))
        elif turn_angle_deg < 0:
            self.env.send(Turn(-1))

        # TODO(marcus): add a step mode option here!

    # TODO(fishberg) THIS DOES NOT SEEM TO WORK AT THE MOMENT?
    def cmd_cb_twist(self, msg):
        """ Listens to teleop force commands and sends to simulator.

            Subscribed to the key_teleop node's output, the callback gets
            Twist messages for both linear and angular 'velocities'. These
            velocities are sent directly as forces and torques to the
            simulator. Actual values for the incoming velocities are
            determined in the launch file.

            Args:
                msg: A geometry_msgs/Twist message.
        """
        force_x = msg.linear.x
        force_y = msg.linear.y
        torque_z = msg.angular.z

        if self.step_mode_enabled:
            self.last_step_cmd.append([force_x, force_y, torque_z])
        else:
            self.env.send(AddForce(force_z=force_x,
                                   torque_y=torque_z,
                                   force_x=force_y))

    def reposition_cb(self, msg):
        """ Listens to pose requests and sends the reposition command
            to the simulator.

            Messages sent in any frame other than the world frame will be
            transformed to the world frame before being sent to the simulator.

            Args:
                msg: A geometry_msgs/PoseStamped message.
        """
        pose = msg.pose.pose
        if msg.header.frame_id != self.world_frame_id:
            pst = PoseStamped()
            pst.pose = msg.pose.pose
            pst.header = msg.header
            try:
                transformed = self.tf_buffer.transform(pst, self.world_frame_id, timeout_sec=0.1)
                pose = transformed.pose
            except Exception:
                # If transform fails, keep original pose
                self.get_logger().warning("Transform for reposition failed; using incoming pose")

        # Hacky frame change from ROS to Unity (left handed)
        # TODO(marcus): We really want to convert to numpy 4x4 and compose!
        py = pose.position.y
        pz = pose.position.z
        qx = pose.orientation.x
        qy = pose.orientation.y
        qz = pose.orientation.z
        qw = pose.orientation.w

        # Rotate 90 degrees around y in the most lazy way possible...
        quat = tf_transformations.quaternion_multiply([-qx, -qz, -qy, qw],
                                                      [0, 0.7071068, 0, 0.7071068])

        pose.position.y = -pz
        pose.position.z = py
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]

        self.reposition_agent(pose)

    def meta_cb(self, data):
        """ Callback for UDP metadata at high rates.

            Parses raw metadata from the simulator, processes it into the
            proper reference frame, and publishes it as odometry, imu and
            transform information to ROS.

            Args:
                data: A string or bytestring in xml format containing the
                    metadata from the simulator.
        """
        # Publish raw metadata.
        if self.publish_metadata:
                self.metadata_pub.publish(msg)

        # Parse metadata and process for proper use.
        metadata = tesse_ros_bridge.utils.parse_metadata(data)
        assert(self.prev_time < metadata['time'])

        metadata_processed = tesse_ros_bridge.utils.process_metadata(metadata,
                self.prev_time, self.prev_vel_brh, self.prev_enu_R_brh)

        timestamp = from_sec_float(metadata_processed['time'])

        # Publish agent ground truth transform.
        self.publish_tf(metadata_processed['transform'], timestamp)

        # Publish clean imu and odometry messages.
        if self.publish_imu:
            if self.use_gt_frames:
                imu = tesse_ros_bridge.utils.metadata_to_imu(metadata_processed,
                        timestamp, self.body_frame_id_gt)
            else:
                imu = tesse_ros_bridge.utils.metadata_to_imu(metadata_processed,
                        timestamp, self.body_frame_id)
            self.clean_imu_pub.publish(imu)

        if self.publish_odom:
            odom = tesse_ros_bridge.utils.metadata_to_odom(metadata_processed,
                    timestamp, self.world_frame_id, self.body_frame_id_gt)
            self.odom_pub.publish(odom)

        # Publish noisy imu and odometry messages:
        if self.publish_noisy_imu or self.publish_noisy_odom:
            metadata_noisy = self.noise_simulator.apply_noise_to_metadata(metadata_processed)

            if self.publish_noisy_imu:
                if self.use_gt_frames:
                    imu = tesse_ros_bridge.utils.metadata_to_imu(metadata_noisy,
                            timestamp, self.body_frame_id_gt)
                else:
                    imu = tesse_ros_bridge.utils.metadata_to_imu(metadata_noisy,
                            timestamp, self.body_frame_id)
                self.noisy_imu_pub.publish(imu)

                # Publish imu biases for debugging:
                if self.publish_imu_noise_biases:
                    self.imu_gyro_bias_vec.header.stamp = timestamp
                    self.imu_gyro_bias_vec.vector.x = self.noise_simulator.gyroscope_bias[0]
                    self.imu_gyro_bias_vec.vector.y = self.noise_simulator.gyroscope_bias[1]
                    self.imu_gyro_bias_vec.vector.z = self.noise_simulator.gyroscope_bias[2]
                    self.imu_gyro_bias_pub.publish(self.imu_gyro_bias_vec)

                    self.imu_accel_bias_vec.header.stamp = timestamp
                    self.imu_accel_bias_vec.vector.x = self.noise_simulator.accelerometer_bias[0]
                    self.imu_accel_bias_vec.vector.y = self.noise_simulator.accelerometer_bias[1]
                    self.imu_accel_bias_vec.vector.z = self.noise_simulator.accelerometer_bias[2]
                    self.imu_accel_bias_pub.publish(self.imu_accel_bias_vec)

            if self.publish_noisy_odom:
                odom = tesse_ros_bridge.utils.metadata_to_odom(metadata_noisy,
                        timestamp, self.world_frame_id, self.body_frame_id_gt)
                self.noisy_odom_pub.publish(odom)

        self.prev_time      = metadata_processed['time']
        self.prev_vel_brh   = metadata_processed['velocity']
        self.prev_enu_R_brh = metadata_processed['transform'][:3,:3]

    def image_cb(self, event=None):
        """ Publish images from simulator to ROS.

            Left and right images are published in the requested encoding.
            Depth images are pre-processed s.t. pixel values directly give
            point depth, in meters.
            Segmentation images are published in the rgb8 encoding.

            Args:
                event: a dummy parameter (kept for compatibility)
        """
        try:
            # Get camera data.
            data_response = self.env.request(DataRequest(True, self.cameras))

            # Process metadata to publish transform.
            metadata = tesse_ros_bridge.utils.parse_metadata(
                data_response.metadata)

            timestamp = from_sec_float(metadata['time'])

            if timestamp == self.last_image_timestamp:
                self.get_logger().info(f"Skipping duplicate images at timestamp {self.last_image_timestamp}")
                return

            # Process each image.
            for i in range(len(self.cameras)):
                if self.cameras[i][0] == Camera.DEPTH:
                    far_draw_dist = self.cam_params[i]['draw_distance']['far']
                    img_msg = self.cv_bridge.cv2_to_imgmsg(
                        data_response.images[i] * far_draw_dist,
                            'passthrough')
                elif self.cameras[i][2] == Channels.SINGLE:
                    img_msg = self.cv_bridge.cv2_to_imgmsg(
                        data_response.images[i], 'mono8')
                elif self.cameras[i][2] == Channels.THREE:
                    img_msg = self.cv_bridge.cv2_to_imgmsg(
                        data_response.images[i], 'rgb8')

                # Sanity check resolutions.
                if self.cam_info_msgs[i] and self.cameras[i][0] != Camera.THIRD_PERSON:
                    assert(img_msg.width == self.cam_info_msgs[i].width)
                    assert(img_msg.height == self.cam_info_msgs[i].height)

                # Publish images to appropriate topic.
                img_msg.header.frame_id = self.cameras[i][3]
                img_msg.header.stamp = timestamp
                self.img_pubs[i].publish(img_msg)

                # Publish associated CameraInfo message.
                if self.cam_info_msgs[i] and self.cam_info_pubs[i]:
                    self.cam_info_msgs[i].header.stamp = timestamp
                    self.cam_info_pubs[i].publish(self.cam_info_msgs[i])

            self.publish_tf(
                tesse_ros_bridge.utils.get_enu_T_brh(metadata),
                    timestamp)

            self.last_image_timestamp = timestamp

        except Exception as error:
            self.get_logger().error(f"TESSE_ROS_NODE: image_cb error: {error}")

    def object_cb(self,event=None):
        """
        """
        obj_metadata = self.env.request(ObjectsRequest()).metadata
        obj_dict = tesse_ros_bridge.utils.parse_object_data(obj_metadata)

        for id, obj in obj_dict.items():
            assert(type(obj) == dict)
            assert(type(id) == int)

            frame_id = "object_" + str(id)
            self.tf_broadcaster.sendTransform(mk_transform(
                                              obj['position'],
                                              obj['quaternion'],
                                              from_sec_float(obj['time']),
                                              frame_id,
                                              self.world_frame_id))

    def scan_cb_slow(self, event=None):
        """ Received LiDAR data from the simulator using standard udp requests
            and waiting for tcp replies. Publishes to ROS.

            Args:
                event: a dummy parameter (kept for compatibility)
        """
        try:
            data_response = self.env.request(LidarDataRequest(True, [lidar[0] for lidar in self.lidars]))
            self.scan_cb(data_response)
        except Exception as error:
            self.get_logger().error(f"TESSE_ROS_NODE: scan_cb_slow error: {error}")

    def scan_cb(self, data):
        """ Receives LiDAR data from the simulator and publishes to ROS.

            Parses raw metadata from the simulator, processes it into the
            proper reference frame, and publishes it as odometry, imu and
            transform information to ROS.

            Args:
                data: A string or bytestring in xml format containing the
                    metadata from the simulator.
        """
        # Parse metadata and process for proper use.
        metadata = tesse_ros_bridge.utils.parse_metadata(data.metadata)

        timestamp = from_sec_float(metadata['time'])

        # Publish scan messages.
        scan = LaserScan()
        scan.header.stamp = timestamp
        for i in range(len(self.lidars)):
            scan.header.frame_id = self.lidars[i][1]

            # TODO(marcus): this is a hack! for some reason the lidar scans are aligned
            #   with the y-axis in this frame. You must "transform" the scans!
            scan.angle_min = self.lidar_params[i]['parameters']['min_angle'] - np.pi/2.0
            scan.angle_max = self.lidar_params[i]['parameters']['max_angle'] - np.pi/2.0
            scan.angle_increment = self.lidar_params[i]['parameters']['angle_inc']
            scan.scan_time = 1.0 / self.scan_rate  # ideal conditions; not guaranteed
            scan.range_min = 0.0
            scan.range_max = self.lidar_params[i]['parameters']['max_range']
            scan.ranges = data.scans[i]

            self.scan_pubs[i].publish(scan)

    def setup_camera(self, camera_params):
        """ Set the parameters, position, and orientation of one camera
            to those values given in the rosparam server.

            Args:
                camera_params: A dictionary containing all the parameters for
                    a single camera. This includes camera id, frame id,
                    intrinsics, and extrinsics.
        """
        # Determine Unity camera, compression, channels
        camera_type_switcher = {
            "left_cam" : Camera.RGB_LEFT,
            "right_cam" : Camera.RGB_RIGHT,
            "seg_cam" : Camera.SEGMENTATION,
            "depth_cam" : Camera.DEPTH,
            "third_person" : Camera.THIRD_PERSON,
        }
        camera_id = camera_type_switcher[camera_params['camera_id']]

        n_channel_switcher = {
            1: Channels.SINGLE,
            3: Channels.THREE,
        }
        n_channels = n_channel_switcher[camera_params['num_channels']]

        # Depth camera is a special case; unity side is rgb but it's converted to mono
        if camera_id == Camera.DEPTH:
            n_channels = Channels.THREE

        # Uniqueness of camera_id is necessary to prevent extra camera setup
        unique_camera_id = True
        for camera in self.cameras:
            if camera[0] == camera_id:
                unique_camera_id = False

        # Store camera object for callbacks
        camera = (camera_id,
                  Compression.ON if camera_params['compression'] else Compression.OFF,
                  n_channels,
                  camera_params['frame_id'],
        )
        self.cameras.append(camera)

        if unique_camera_id:
            width          = camera_params['width']
            height         = camera_params['height']
            vertical_fov   = camera_params['vertical_fov']
            near_draw_dist = camera_params['near_draw_dist']
            far_draw_dist  = camera_params['far_draw_dist']

            pos_x = camera_params['pos_x']
            pos_y = camera_params['pos_y']
            pos_z = camera_params['pos_z']

            quat_x = camera_params['quat_x']
            quat_y = camera_params['quat_y']
            quat_z = camera_params['quat_z']
            quat_w = camera_params['quat_w']

            # Set parameters
            resp = None
            while resp is None:
                self.get_logger().info("TESSE_ROS_NODE: Setting intrinsic parameters for camera: %s" % camera_id)
                resp = self.env.request(SetCameraParametersRequest(
                    camera_id,
                    height,
                    width,
                    vertical_fov,
                    near_draw_dist,
                    far_draw_dist))

            # Set position
            resp = None
            while resp is None:
                self.get_logger().info("TESSE_ROS_NODE: Setting position of camera: %s" % camera_id)
                resp = self.env.request(SetCameraPositionRequest(
                        camera_id,
                        pos_x,
                        pos_y,
                        pos_z,))

            # Set orientation
            while resp is None:
                self.get_logger().info("TESSE_ROS_NODE: Setting orientation of camera: %s" % camera_id)
                resp = self.env.request(SetCameraOrientationRequest(
                        camera_id,
                        quat_x,
                        quat_y,
                        quat_z,
                        quat_w,))

            # Get information back from simulator
            cam_data = None
            while cam_data is None:
                self.get_logger().info("TESSE_ROS_NODE: Acquiring camera data for camera: %s" % camera_id)
                cam_data = tesse_ros_bridge.utils.parse_cam_data(
                    self.env.request(
                        CameraInformationRequest(camera_id)).metadata)

            assert(cam_data['id'] == camera_id.value)
            assert(cam_data['parameters']['height'] == height)
            assert(cam_data['parameters']['width'] == width)
            self.cam_params.append(cam_data)

            # Store static transform for camera
            static_tf_cam                       = TransformStamped()
            static_tf_cam.header.frame_id       = self.body_frame_id
            if self.use_gt_frames:
                static_tf_cam.header.frame_id   = self.body_frame_id_gt
            static_tf_cam.child_frame_id        = camera[3]
            static_tf_cam.header.stamp          = self.get_clock().now().to_msg()
            static_tf_cam.transform.translation = Point(cam_data['position'][0],
                                                        cam_data['position'][1],
                                                        cam_data['position'][2])
            static_tf_cam.transform.rotation    = Quaternion(cam_data['quaternion'][0],
                                                             cam_data['quaternion'][1],
                                                             cam_data['quaternion'][2],
                                                             cam_data['quaternion'][3])

            self.static_tfs_to_broadcast.append(static_tf_cam)

            # Store camera information message for publishing
            cam_info_msg = tesse_ros_bridge.utils.generate_camera_info(
                    cam_data, camera[3])

            # Initialize the publisher for the camera info
            cam_info_pub = self.create_publisher(CameraInfo, camera_params['camera_id'] + "/camera_info", qos_profile=qos10)

            self.cam_info_msgs.append(cam_info_msg)
            self.cam_info_pubs.append(cam_info_pub)

        else:
            # We have to append something to these state vars to keep lengths correct
            self.cam_params.append(None)
            self.cam_info_msgs.append(None)
            self.cam_info_pubs.append(None)

        # Initialize the publisher for the image
        if n_channel_switcher[camera_params['num_channels']] == Channels.THREE:
            self.img_pubs.append(self.create_publisher(ImageMsg, camera_params['camera_id'] + "/rgb/image_raw", qos_profile=qos10))
        elif n_channel_switcher[camera_params['num_channels']] == Channels.SINGLE:
            self.img_pubs.append(self.create_publisher(ImageMsg, camera_params['camera_id'] + "/mono/image_raw", qos_profile=qos10))

    def setup_all_cameras(self):
        """ Sets up all cameras based on the publish flags passed from launch. """
        camera_params = self.declare_parameter("camera_params", 'NOT SET').value
        if camera_params == 'NOT SET':
            camera_params = {}

        if self.publish_stereo_rgb:
            self.setup_camera(camera_params['RGB_LEFT'])
            self.setup_camera(camera_params['RGB_RIGHT'])

        if self.publish_stereo_gry:
            self.setup_camera(camera_params['GRY_LEFT'])
            self.setup_camera(camera_params['GRY_RIGHT'])

        if self.publish_segmentation:
            self.setup_camera(camera_params['SEGMENTATION'])

        if self.publish_depth:
            self.setup_camera(camera_params['DEPTH'])

        if self.publish_third_pov:
            self.setup_camera(camera_params['THIRD_PERSON'])

    def setup_lidar(self, lidar_params):
        """ Set up one LiDAR in the simulator based on the parameters given
            by the rosparam server.

            Args:
                lidar_params: A dictionary containing the parameters for a single
                    lidar sensor. This includes camera id, frame id, pose, scan angles,
                    max range, and number of beams.
        """
        lidar_type_switcher = {
            "front_lidar" : Lidar.HOOD,
            "rear_lidar" : Lidar.TRUNK,
        }
        lidar_id = lidar_type_switcher[lidar_params['lidar_id']]

        lidar = (lidar_id, lidar_params['frame_id'])
        self.lidars.append(lidar)

        # Get all lidar parameters
        min_angle = lidar_params["scan_min_angle"]
        max_angle = lidar_params["scan_max_angle"]
        max_range = lidar_params["scan_max_range"]
        num_beams = lidar_params["scan_beams"]

        pos_x = lidar_params["pos_x"]
        pos_y = lidar_params["pos_y"]
        pos_z = lidar_params["pos_z"]

        quat_x = lidar_params["quat_x"]
        quat_y = lidar_params["quat_y"]
        quat_z = lidar_params["quat_z"]
        quat_w = lidar_params["quat_w"]

        # Set parameters
        resp = None
        while resp is None:
            self.get_logger().info("TESSE_ROS_NODE: Setting intrinsic parameters for lidar: %s" % lidar_id)
            resp = self.env.request(SetLidarParametersRequest(
                lidar_id,
                min_angle=min_angle,
                max_angle=max_angle,
                max_range=max_range,
                ray_count=num_beams))

        # Set position
        if pos_x != "default" or pos_y != "default" or pos_z != "default":
            resp = None
            while resp is None:
                self.get_logger().info("TESSE_ROS_NODE: Setting position of lidar: %s" % lidar_id)
                resp = self.env.request(SetLidarPositionRequest(
                        lidar_id,
                        pos_x,
                        pos_y,
                        pos_z))

        # Set orientation
        if quat_x != "default" or quat_y != "default" or \
                quat_z != "default" or quat_w != "default":
            resp = None
            while resp is None:
                self.get_logger().info("TESSE_ROS_NODE: Setting orientation of lidar: %s" % lidar_id)
                resp = self.env.request(SetLidarOrientationRequest(
                        lidar_id,
                        quat_x,
                        quat_y,
                        quat_z,
                        quat_w))

        # Get information back from simulator
        lidar_data = None
        while lidar_data is None:
            self.get_logger().info("TESSE_ROS_NODE: Acquiring lidar data for lidar: %s" % lidar_id)
            lidar_data = tesse_ros_bridge.utils.parse_lidar_data(
                self.env.request(LidarInformationRequest(lidar_id)).metadata)

            tol = 1e-6
            assert(lidar_data["id"] == lidar_id.value)
            assert(abs(lidar_data['parameters']["min_angle"] - min_angle) < tol)
            assert(abs(lidar_data['parameters']["max_angle"] - max_angle) < tol)
            assert(abs(lidar_data['parameters']["max_range"] - max_range) < tol)
            assert(abs(lidar_data['parameters']["ray_count"] - num_beams) < tol)

        # Publish static transform for lidar
        scan_ts                       = TransformStamped()
        scan_ts.header.stamp          = self.get_clock().now().to_msg()
        scan_ts.header.frame_id       = self.body_frame_id
        if self.use_gt_frames:
            scan_ts.header.frame_id   = self.body_frame_id_gt
        scan_ts.child_frame_id = lidar[1]
        scan_ts.transform.translation = Point(lidar_data['position'][0],
                                              lidar_data['position'][1],
                                              lidar_data['position'][2])
        scan_ts.transform.rotation = Quaternion(lidar_data['quaternion'][0],
                                                lidar_data['quaternion'][1],
                                                lidar_data['quaternion'][2],
                                                lidar_data['quaternion'][3])

        self.static_tfs_to_broadcast.append(scan_ts)

        self.lidar_params.append(lidar_data)

        # Set up scan publisher
        self.scan_pubs.append(self.create_publisher(LaserScan, lidar[1] + "/scan", qos_profile=qos10))

    def setup_all_lidars(self):
        """ Sets up all LiDARs in the simulator based on the publish flags
            passed in from launch.
        """
        lidar_params = self.declare_parameter("lidar_params", 'NOT SET').value
        if lidar_params == 'NOT SET':
            lidar_params = {}
        # Get all lidars to be used
        if self.publish_front_lidar:
            self.setup_lidar(lidar_params['FRONT'])

        if self.publish_rear_lidar:
            self.setup_lidar(lidar_params['REAR'])
    
    # TODO(marcus): is this useful for general purpose?
    # def setup_static_map_tf(self):
    #     """
    #     """
    #     print("TESSE_ROS_NODE: Setting up static transform to map", )
    #     scene_switcher = {
    #         0: "what_are_you_doing_this_is_a_fake_scene",
    #         1: "windridge_city",
    #         2: "test_scene",
    #     }

    #     initial_scene = rospy.get_param("~initial_scene")
    #     tf_dict = rospy.get_param("~" + scene_switcher[initial_scene] + "_tf")

    #     world_T_map = TransformStamped()
    #     world_T_map.header.stamp = rospy.Time.now()
    #     world_T_map.header.frame_id = self.world_frame_id
    #     world_T_map.child_frame_id = self.map_frame_id
    #     world_T_map.transform.translation = Point(tf_dict['px'],
    #                                               tf_dict['py'],
    #                                               tf_dict['pz'])
    #     world_T_map.transform.rotation = Quaternion(tf_dict['qx'],
    #                                                 tf_dict['qy'],
    #                                                 tf_dict['qz'],
    #                                                 tf_dict['qw'])

    #     self.static_tfs_to_broadcast.append(world_T_map)

    def spawn_initial_objects(self):
        """ Spawn initial objects from the parameter yaml file. """
        self.get_logger().info("TESSE_ROS_NODE: Spawning initial objects")
        num_objects = int(self.declare_parameter("num_objects", 0).value)

        for i in range(num_objects):
            obj_dict = self.declare_parameter(f"object_{i}", {}).value
            pose = Pose()
            params = []
            if obj_dict.get('use_custom_pose', False):
                pose.position.x = obj_dict['px']
                pose.position.y = obj_dict['py']
                pose.position.z = obj_dict['pz']
                pose.orientation.x = obj_dict['qx']
                pose.orientation.y = obj_dict['qy']
                pose.orientation.z = obj_dict['qz']
                pose.orientation.w = obj_dict['qw']
            if obj_dict.get('send_params', False):
                if 'params' in obj_dict.keys():
                    for param in obj_dict['params']:
                        params.append(param)
                else:
                    # Choose random parameters.
                    # TODO(marcus): this is specific to SMPL! Generalize the number of params.
                    params = [(np.random.random() * 2) - 1 for _ in range(10)]

            self.spawn_object(obj_dict['id'], pose, params)

    def setup_collision(self, enable_collision):
        """ Enable/Disable collisions in Simulator. """
        self.get_logger().info(f"TESSE_ROS_NODE: Setup collisions to: {enable_collision}")
        if enable_collision is True:
            self.env.send(ColliderRequest(enable=1))
        else:
            self.env.send(ColliderRequest(enable=0))

    def setup_ros_services(self):
        """ Setup ROS services related to the simulator.

            These services include:
                scene_change_request: change the scene_id of the simulator
                object_spawn_request: spawn a prefab object into the scene
                reposition_request:   reposition the agent to a desired pose
        """
        self.scene_request_service = self.create_service(SceneRequestService, "scene_change_request", self.rosservice_change_scene)
        self.change_scene_client = self.create_client(SceneRequestService, 'scene_change_request')

        self.object_spawn_service = self.create_service(ObjectSpawnRequestService, "object_spawn_request", self.rosservice_spawn_object)
        self.spawn_object_client = self.create_client(ObjectSpawnRequestService, 'object_spawn_request')

        self.reposition_request_service = self.create_service(RepositionRequestService, 'reposition_request', self.rosservice_reposition)
        self.reposition_agent_client = self.create_client(RepositionRequestService, 'reposition_request')

    def rosservice_change_scene(self, request, response):
        """ Change scene ID of simulator as a ROS service. """
        try:
            self.env.request(SceneRequest(request.id))
            return response
        except Exception as e:
            self.get_logger().error(f"Scene Change Error: {e}")
            return response

    def rosservice_spawn_object(self, request, response):
        """ Spawn an object into the simulator as a ROS service. """
        try:
            if request.pose == Pose():
                self.env.request(SpawnObjectRequest(object_index=request.id,
                                                    method=ObjectSpawnMethod.RANDOM,
                                                    params=request.params))
            else:
                self.env.request(SpawnObjectRequest(object_index=request.id,
                                                    method=ObjectSpawnMethod.USER,
                                                    position_x=request.pose.position.x,
                                                    position_y=request.pose.position.y,
                                                    position_z=request.pose.position.z,
                                                    orientation_x=request.pose.orientation.x,
                                                    orientation_y=request.pose.orientation.y,
                                                    orientation_z=request.pose.orientation.z,
                                                    orientation_w=request.pose.orientation.w,
                                                    params=request.params))
            return response
        except Exception as e:
            self.get_logger().error(f"Object Spawn Error: {e}")
            return response

    def rosservice_reposition(self, request, response):
        """ Repositions the agent to a desired pose as a ROS service. """
        try:
            self.env.send(Reposition(request.pose.position.x,
                          request.pose.position.y,
                          request.pose.position.z,
                          request.pose.orientation.x,
                          request.pose.orientation.y,
                          request.pose.orientation.z,
                          request.pose.orientation.w))
            return response
        except Exception as e:
            self.get_logger().error(f"Reposition Error: {e}")
            return response

    def publish_tf(self, cur_tf, timestamp):
        """ Publish the ground-truth transform to the TF tree.

            Args:
                cur_tf: A 4x4 numpy matrix containing the transformation from
                    the body frame of the agent to ENU.
                timestamp: A rospy.Time instance representing the current
                    time in the simulator.
        """
        # Publish current transform to tf tree.
        trans = tesse_ros_bridge.utils.get_translation_part(cur_tf)
        quat = tesse_ros_bridge.utils.get_quaternion(cur_tf)
        self.tf_broadcaster.sendTransform(mk_transform(
                                          trans, quat, timestamp,
                                          self.body_frame_id_gt,
                                          self.world_frame_id))


def main(args=None):
    rclpy.init(args=args)
    node = TesseROSWrapper()
    node.spin() # create timers
    try:
        rclpy.spin(node)  # control timers y callbacks
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

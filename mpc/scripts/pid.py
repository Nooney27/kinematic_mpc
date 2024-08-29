#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from utils import nearest_point
from dataclasses import dataclass, field

@dataclass
class PIDConfig:
    Kp: float = 5.0
    Ki: float = 0.0
    Kd: float = 0.1
    max_steering_angle: float = 0.4189  # 24 degrees in radians
    max_speed: float = 12.0  # maximum speed [m/s]
    dt: float = 0.1  # time step

class PIDState:
    def __init__(self):
        self.integral_error = 0.0
        self.prev_error = 0.0

@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    v: float = 0.0
    yaw: float = 0.0

class PIDController(Node):
    def __init__(self):
        super().__init__('pid_controller_node')
        self.is_real = False  # Switch between simulator and real car topics
        self.config = PIDConfig()
        self.state = PIDState()

        self.map_name = 'interpolated_trajectory'
        self.waypoints = np.loadtxt(self.map_name + '.csv', delimiter=';', skiprows=1)
        print('Loaded waypoints: ', self.waypoints)
        self.waypoints[:, 3] += math.pi / 2 

        drive_topic = '/drive'
        odom_topic = '/pf/viz/inferred_pose' if self.is_real else '/ego_racecar/odom'
        ref_path_tracker = '/ref_path_tracker'

        # Pose subscriber
        self.sub_pose = self.create_subscription(PoseStamped if self.is_real else Odometry, odom_topic, self.pose_callback, 1)
        # Drive publisher
        self.pub_drive = self.create_publisher(AckermannDriveStamped, drive_topic, 1)
        self.drive_msg = AckermannDriveStamped()
        # Visualization of reference path publisher
        self.ref_path_vis = self.create_publisher(Marker, ref_path_tracker, 1)
        self.ref_path_msg = Marker()

        self.visualize_waypoints()

    def pose_callback(self, pose_msg):
        vehicle_state = self.get_vehicle_state(pose_msg)
        target_x, target_y, target_yaw, target_v = self.calc_target(vehicle_state)

        steering_output, speed_output = self.pid_control(vehicle_state, target_x, target_y, target_yaw, target_v)

        self.drive_msg.drive.steering_angle = steering_output
        self.drive_msg.drive.speed = speed_output
        print(f"Steering: {steering_output}, Speed: {speed_output}")
        self.pub_drive.publish(self.drive_msg)


    def pid_control(self, state, target_x, target_y, target_yaw, target_v):
        #error_yaw = target_yaw - state.yaw
        error_x = target_x - state.x
        error_y = target_y - state.y
        
        # Assuming we use the y-error to control steering
        steering_output = (
            self.config.Kp * error_y +
            self.config.Ki * self.state.integral_error +
            self.config.Kd * ((error_y - self.state.prev_error) / self.config.dt)
        )
        steering_output = max(-self.config.max_steering_angle, min(self.config.max_steering_angle, steering_output))

        # Speed control based on distance to the target point
        distance_to_target = math.sqrt(error_x**2 + error_y**2)
        speed_output = max(0.0, min(self.config.max_speed, distance_to_target))

        self.state.integral_error += error_y * self.config.dt
        self.state.prev_error = error_y
        '''
        self.state.integral_error += error_yaw * self.config.dt
        derivative_error = (error_yaw - self.state.prev_error) / self.config.dt

        steering_output = (
            self.config.Kp * error_yaw +
            self.config.Ki * self.state.integral_error +
            self.config.Kd * derivative_error
        )
        steering_output = max(-self.config.max_steering_angle, min(self.config.max_steering_angle, steering_output))

        speed_output = max(0.0, min(self.config.max_speed, target_v))

        self.state.prev_error = error_yaw'''

        return steering_output, speed_output

    def get_vehicle_state(self, pose_msg):
        vehicle_state = State()
        vehicle_state.x = pose_msg.pose.position.x if self.is_real else pose_msg.pose.pose.position.x
        vehicle_state.y = pose_msg.pose.position.y if self.is_real else pose_msg.pose.pose.position.y
        vehicle_state.v = self.drive_msg.drive.speed
        quat_msg = pose_msg.pose.orientation if self.is_real else pose_msg.pose.pose.orientation
        quat = [quat_msg.x, quat_msg.y, quat_msg.z, quat_msg.w]
        vehicle_state.yaw = math.atan2(2 * (quat[3] * quat[2] + quat[0] * quat[1]), 1 - 2 * (quat[1] ** 2 + quat[2] ** 2))
        return vehicle_state

    def calc_target(self, state):
        _, _, _, ind = nearest_point(np.array([state.x, state.y]), np.array([self.waypoints[:, 1], self.waypoints[:, 2]]).T)
        target_x = self.waypoints[ind, 1]
        target_y = self.waypoints[ind, 2]
        target_yaw = self.waypoints[ind, 3]
        target_v = self.waypoints[ind, 5]
        return target_x, target_y, target_yaw, target_v

    def visualize_waypoints(self):
        waypoints_msg = Marker()
        waypoints_msg.points = []
        waypoints_msg.header.frame_id = '/map'
        waypoints_msg.type = Marker.POINTS
        waypoints_msg.color.g = 0.75
        waypoints_msg.color.a = 1.0
        waypoints_msg.scale.x = 0.05
        waypoints_msg.scale.y = 0.05
        waypoints_msg.id = 0
        for i in range(self.waypoints.shape[0]):
            point = Point(x=self.waypoints[i, 1], y=self.waypoints[i, 2], z=0.1)
            waypoints_msg.points.append(point)
        self.ref_path_vis.publish(waypoints_msg)

def main(args=None):
    rclpy.init(args=args)
    print("PID Controller Initialized")
    pid_node = PIDController()
    rclpy.spin(pid_node)

    pid_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

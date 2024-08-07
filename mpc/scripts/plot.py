import math
import os
from dataclasses import dataclass, field
import sys
import cvxpy
import numpy as np
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import PoseStamped, Point, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix, diags
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from utils import nearest_point

@dataclass
class mpc_config:
    # [existing config parameters]
    pass

@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    delta: float = 0.0
    v: float = 0.0
    yaw: float = 0.0
    yawrate: float = 0.0
    beta: float = 0.0

class MPC(Node):
    def __init__(self):
        super().__init__('mpc_node')
        
        self.real_car = False

        self.map_name = 'traj_race_cl(2)'  # Ensure this path is correct
        self.waypoints = np.loadtxt(self.map_name + '.csv', delimiter=';', skiprows=1) 
        print('waypoints: ', self.waypoints)
        
        #self.waypoints[:, 4] += math.pi 

        drive_topic = '/drive'
        if self.real_car:
            odom_topic = '/pf/viz/inferred_pose'
        else:
            odom_topic = '/ego_racecar/odom'

        self.pub_drive = self.create_publisher(AckermannDriveStamped, drive_topic, 1)
        self.pub_marker = self.create_publisher(MarkerArray, 'visualization_marker_array', 10)
        
        self.drive_msg = AckermannDriveStamped()        
        self.config = mpc_config()
        self.odelta_v = None
        self.odelta = None
        self.oa = None
        self.init_flag = 0

        self.initial_x = None
        self.initial_y = None
        self.initial_yaw = None
        if self.real_car:
            self.load_initial_pose()
        self.mpc_prob_init()
        
        # Publish markers for waypoints
        self.publish_waypoints_as_markers()

    def load_initial_pose(self):
        wp = np.loadtxt(self.map_name + '.csv', delimiter=";", skiprows=0, max_rows=1)
        self.initial_x = wp[1]
        self.initial_y = wp[2]
        print(f"Initial pose: {self.initial_x}, {self.initial_y}")

    def publish_waypoints_as_markers(self):
        marker_array = MarkerArray()
        for i, waypoint in enumerate(self.waypoints):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "waypoints"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = waypoint[1]
            marker.pose.position.y = waypoint[2]
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            marker.color.a = 1.0  # Don't forget to set the alpha!
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)

        self.pub_marker.publish(marker_array)

    def pose_callback(self, pose_msg):
        vehicle_state = self.get_vehicle_state(pose_msg)
        ref_path = self.calc_ref_trajectory(vehicle_state, self.waypoints[:,0], self.waypoints[:,1], self.waypoints[:,3], self.waypoints[:,4])
        print(f"Ref path: {ref_path}")
        x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]
        print(f"x0: {x0}")
        (
            self.oa,
            self.odelta_v,
            ox,
            oy,
            oyaw,
            ov,
            state_predict,
        ) = self.linear_mpc_control(ref_path, x0, self.oa, self.odelta_v)

        steer_output = self.odelta_v[0]
        speed_output = vehicle_state.v + self.oa[0] * self.config.DTK
        if speed_output < 0.0:
            speed_output = 0.5
        self.drive_msg.drive.steering_angle = steer_output
        self.drive_msg.drive.speed = speed_output
        print(f"Steering: {steer_output}, Speed: {speed_output}")
        self.pub_drive.publish(self.drive_msg)

    def get_vehicle_state(self, pose_msg):
        vehicle_state = State()
        if self.real_car:
            vehicle_state.x = (pose_msg.pose.pose.position.x - self.initial_x)
            vehicle_state.y = (pose_msg.pose.pose.position.y - self.initial_y)
            print(f"X: {vehicle_state.x}, Y: {vehicle_state.y}")
            print(f"position x: {pose_msg.pose.position.x}, position y: {pose_msg.pose.position.y}")
            quat_msg = pose_msg.pose.orientation
            print(f"quat_msg: {quat_msg}")
        else:
            vehicle_state.x = pose_msg.pose.pose.position.x
            vehicle_state.y = pose_msg.pose.pose.position.y
            quat_msg = pose_msg.pose.pose.orientation
            print(f"X: {vehicle_state.x}, Y: {vehicle_state.y}")
            print(f"position x: {vehicle_state.x}, position y: {vehicle_state.y}")
            print(f"quat_msg: {quat_msg}")

        vehicle_state.v = self.drive_msg.drive.speed
        quat = [quat_msg.x, quat_msg.y, quat_msg.z, quat_msg.w]
        vehicle_state.yaw = math.atan2(2 * (quat[3] * quat[2] + quat[0] * quat[1]), 1 - 2 * (quat[1] ** 2 + quat[2] ** 2))
        return vehicle_state

    def mpc_prob_init(self):
        # [Your existing MPC initialization code]
        pass

    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp):
        # [Your existing trajectory calculation code]
        pass

    def predict_motion(self, x0, oa, od, xref):
        # [Your existing motion prediction code]
        pass

    def update_state(self, state, a, delta):
        # [Your existing state update code]
        pass

    def get_model_matrix(self, v, phi, delta):
        # [Your existing model matrix calculation code]
        pass

    def mpc_prob_solve(self, ref_traj, path_predict, x0):
        # [Your existing MPC problem solving code]
        pass

    def linear_mpc_control(self, ref_path, x0, oa, od):
        # [Your existing MPC control code]
        pass

def main(args=None):
    rclpy.init(args=args)
    print("MPC Initialized")
    mpc_node = MPC()
    rclpy.spin(mpc_node)

    mpc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


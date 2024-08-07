import math
import os
from dataclasses import dataclass, field
import cvxpy
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix, diags
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from utils import nearest_point
from PyQt5.QtWidgets import QApplication
from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
import threading


@dataclass
class mpc_config:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
    NU: int = 2  # length of input vector: u = [steering speed, acceleration]
    TK: int = 6  # finite time horizon length - kinematic

    # TODO: you may need to tune the following matrices
    Rk: list = field(
        default_factory=lambda: np.diag([0.01, 100.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.01, 100.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: list = field(
        default_factory=lambda: np.diag([100.0, 100.0, 8.0, 13.0])  
    )  # state error cost matrix, for the the next (T) prediction time steps [x, y, delta, v, yaw, yaw-rate, beta]
    Qfk: list = field(
        default_factory=lambda: np.diag([5.0, 5.0, 5.5, 13.0])
    )  # final state error matrix, penalty  for the final state constraints: [x, y, delta, v, yaw, yaw-rate, beta]

    N_IND_SEARCH: int = 20
    DTK: float = 0.1
    dlk: float = 0.2
    LENGTH: float = 0.9
    WIDTH: float = 0.54
    WB: float = 0.53
    MIN_STEER: float = -0.4189
    MAX_STEER: float = 0.4189
    MAX_DSTEER: float = np.deg2rad(180.0)
    MAX_STEER_V: float = 3.2
    MAX_SPEED: float = 8.0
    MIN_SPEED: float = 0.0
    MAX_ACCEL: float = 3.0

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

        self.real_car = True
        self.config_path = "/home/nvidia/pure_pursuit_results"
        self.csv = "traj_race_cl(2).csv"
        self.map_name = os.path.join(self.config_path, self.csv)
        # x, y, z, v, yaw
        self.waypoints = np.loadtxt(self.map_name, delimiter=';', skiprows=1)
        self.waypoints[:, 3] += math.pi/2

        drive_topic = '/drive'
        if self.real_car:
            odom_topic = '/pf/viz/inferred_pose'
        else:
            odom_topic = '/ego_racecar/odom'

        if self.real_car:
            self.sub_pose = self.create_subscription(PoseWithCovarianceStamped, '/gnss_to_local/local_position', self.pose_callback, 10)
        else:
            self.sub_pose = self.create_subscription(Odometry, odom_topic, self.pose_callback, 10)

        self.pub_drive = self.create_publisher(AckermannDriveStamped, drive_topic, 1)
        self.drive_msg = AckermannDriveStamped()
        self.config = mpc_config()
        self.odelta_v = None
        self.odelta = None
        self.oa = None
        self.init_flag = 0

        self.initial_x = None
        self.initial_y = None
        self.initial_yaw = None
        """ if self.real_car:
            self.load_initial_pose() """
        self.mpc_prob_init()

        # Initialize PyQtGraph
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.app = QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=True, title="MPC Position Tracking")
        self.plot = self.win.addPlot(title="Trajectory and Waypoints")
        self.plot.enableAutoRange('xy', True)
        self.plot.setAspectLocked(True)

        self.waypoints_plot = pg.ScatterPlotItem(size=5, brush=pg.mkBrush(0, 0, 255), name="Waypoints")
        self.trajectory_plot = pg.PlotCurveItem(pen=pg.mkPen('r', width=2), name="Trajectory")
        self.current_location_plot = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(0, 255, 0), name="Current Location")

        self.plot.addItem(self.waypoints_plot)
        self.plot.addItem(self.trajectory_plot)
        self.plot.addItem(self.current_location_plot)

        self.waypoints_plot.setData([{'pos': (wp[1], wp[2]), 'data': 1} for wp in self.waypoints])

        self.points = []
        self.current_point = []

        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.update_plot)
        self.update_timer.start(50)

        self.legend = self.plot.addLegend()
        self.legend.addItem(self.waypoints_plot, 'Waypoints')
        self.legend.addItem(self.trajectory_plot, 'Trajectory')
        self.legend.addItem(self.current_location_plot, 'Current Position')

    def load_initial_pose(self):
        wp = np.loadtxt(self.map_name, delimiter=";", skiprows=0, max_rows=1)
        self.initial_x = wp[0]
        self.initial_y = wp[1]
        print(f"Initial pose: {self.initial_x}, {self.initial_y}")

    def pose_callback(self, pose_msg):
        vehicle_state = self.get_vehicle_state(pose_msg)
        ref_path = self.calc_ref_trajectory(vehicle_state, self.waypoints[:, 1], self.waypoints[:, 2], self.waypoints[:,3], self.waypoints[:, 5])
        x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]
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
        self.pub_drive.publish(self.drive_msg)

        self.points.append((vehicle_state.x, vehicle_state.y))
        self.current_point = [(vehicle_state.x, vehicle_state.y)]

    def get_vehicle_state(self, pose_msg):
        vehicle_state = State()

        if self.real_car:
            vehicle_state.x = (pose_msg.pose.pose.position.x)# - self.initial_x)
            vehicle_state.y = (pose_msg.pose.pose.position.y)# - self.initial_y)
            quat_msg = pose_msg.pose.pose.orientation
        else:
            vehicle_state.x = pose_msg.pose.pose.position.x
            vehicle_state.y = pose_msg.pose.pose.position.y
            quat_msg = pose_msg.pose.pose.orientation

        vehicle_state.v = self.drive_msg.drive.speed
        quat = [quat_msg.x, quat_msg.y, quat_msg.z, quat_msg.w]
        vehicle_state.yaw = math.atan2(2 * (quat[3] * quat[2] + quat[0] * quat[1]), 1 - 2 * (quat[1] ** 2 + quat[2] ** 2))
        #vehicle_state.yaw += math.pi
        # vehicle_state.yaw = 0
        return vehicle_state

    def mpc_prob_init(self):
        self.xk = cvxpy.Variable(
            (self.config.NXK, self.config.TK + 1)
        )
        self.uk = cvxpy.Variable(
            (self.config.NU, self.config.TK)
        )
        objective = 0.0
        constraints = []

        self.x0k = cvxpy.Parameter((self.config.NXK,))
        self.x0k.value = np.zeros((self.config.NXK,))

        self.ref_traj_k = cvxpy.Parameter((self.config.NXK, self.config.TK + 1))
        self.ref_traj_k.value = np.zeros((self.config.NXK, self.config.TK + 1))

        R_block = block_diag(tuple([self.config.Rk] * self.config.TK))
        Rd_block = block_diag(tuple([self.config.Rdk] * (self.config.TK - 1)))
        Q_block = [self.config.Qk] * (self.config.TK)
        Q_block.append(self.config.Qfk)
        Q_block = block_diag(tuple(Q_block))

        obj1 = cvxpy.quad_form(cvxpy.vec(self.uk), R_block)
        obj2 = cvxpy.quad_form(cvxpy.vec(self.xk - self.ref_traj_k), Q_block)
        obj3 = cvxpy.quad_form(cvxpy.vec(cvxpy.diff(self.uk, axis=1)), Rd_block)
        objective = obj1 + obj2 + obj3

        A_block = []
        B_block = []
        C_block = []
        path_predict = np.zeros((self.config.NXK, self.config.TK + 1))
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], 0.0
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        m, n = A_block.shape
        self.Annz_k = cvxpy.Parameter(A_block.nnz)
        data = np.ones(self.Annz_k.size)
        rows = A_block.row * n + A_block.col
        cols = np.arange(self.Annz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Annz_k.size))

        self.Annz_k.value = A_block.data
        self.Ak_ = cvxpy.reshape(Indexer @ self.Annz_k, (m, n), order="C")

        m, n = B_block.shape
        self.Bnnz_k = cvxpy.Parameter(B_block.nnz)
        data = np.ones(self.Bnnz_k.size)
        rows = B_block.row * n + B_block.col
        cols = np.arange(self.Bnnz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Bnnz_k.size))
        self.Bk_ = cvxpy.reshape(Indexer @ self.Bnnz_k, (m, n), order="C")
        self.Bnnz_k.value = B_block.data

        self.Ck_ = cvxpy.Parameter(C_block.shape)
        self.Ck_.value = C_block

        constraint1 = cvxpy.vec(self.xk[:, 1:]) == self.Ak_ @ cvxpy.vec(self.xk[:, :-1]) + self.Bk_ @ cvxpy.vec(self.uk) + self.Ck_
        constraint2 = cvxpy.abs(self.uk[1, 1:] - self.uk[1, :-1]) <= self.config.MAX_DSTEER * self.config.DTK
        constraint3 = self.x0k == self.xk[:, 0]
        constraint4 = cvxpy.abs(self.xk[2, :]) <= self.config.MAX_SPEED
        constraint5 = cvxpy.abs(self.uk[0, :]) <= self.config.MAX_ACCEL
        constraint6 = cvxpy.abs(self.uk[1, :]) <= self.config.MAX_STEER
        constraints = [constraint1, constraint2, constraint3, constraint4, constraint5, constraint6]

        self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp):
        ref_traj = np.zeros((self.config.NXK, self.config.TK + 1))
        ncourse = len(cx)
        _, _, _, ind = nearest_point(np.array([state.x, state.y]), np.array([cx, cy]).T)
        ref_traj[0, 0] = cx[ind]
        ref_traj[1, 0] = cy[ind]
        ref_traj[2, 0] = sp[ind]
        ref_traj[3, 0] = cyaw[ind]
        travel = abs(state.v) * self.config.DTK
        dind = travel / self.config.dlk
        dind = 2
        ind_list = int(ind) + np.insert(
            np.cumsum(np.repeat(dind, self.config.TK)), 0, 0
        ).astype(int)
        ind_list[ind_list >= ncourse] -= ncourse
        ref_traj[0, :] = cx[ind_list]
        ref_traj[1, :] = cy[ind_list]
        ref_traj[2, :] = sp[ind_list]

        angle_thres = 4.5
        for i in range(len(cyaw)):
            if cyaw[i] - state.yaw > angle_thres:
                cyaw[i] -= 2*np.pi
            if state.yaw - cyaw[i] > angle_thres:
                cyaw[i] += 2*np.pi

        ref_traj[3, :] = cyaw[ind_list]
        return ref_traj

    def predict_motion(self, x0, oa, od, xref):
        path_predict = xref * 0.0
        for i, _ in enumerate(x0):
            path_predict[i, 0] = x0[i]
        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for (ai, di, i) in zip(oa, od, range(1, self.config.TK + 1)):
            state = self.update_state(state, ai, di)
            path_predict[0, i] = state.x
            path_predict[1, i] = state.y
            path_predict[2, i] = state.v
            path_predict[3, i] = state.yaw
        return path_predict

    def update_state(self, state, a, delta):
        if delta >= self.config.MAX_STEER:
            delta = self.config.MAX_STEER
        elif delta <= -self.config.MAX_STEER:
            delta = -self.config.MAX_STEER
        state.x = state.x + state.v * math.cos(state.yaw) * self.config.DTK
        state.y = state.y + state.v * math.sin(state.yaw) * self.config.DTK
        state.yaw = (
            state.yaw + (state.v / self.config.WB) * math.tan(delta) * self.config.DTK
        )
        state.v = state.v + a * self.config.DTK
        if state.v > self.config.MAX_SPEED:
            state.v = self.config.MAX_SPEED
        elif state.v < self.config.MIN_SPEED:
            state.v = self.config.MIN_SPEED
        return state

    def get_model_matrix(self, v, phi, delta):
        A = np.zeros((self.config.NXK, self.config.NXK))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.config.DTK * math.cos(phi)
        A[0, 3] = -self.config.DTK * v * math.sin(phi)
        A[1, 2] = self.config.DTK * math.sin(phi)
        A[1, 3] = self.config.DTK * v * math.cos(phi)
        A[3, 2] = self.config.DTK * math.tan(delta) / self.config.WB

        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = self.config.DTK
        B[3, 1] = self.config.DTK * v / (self.config.WB * math.cos(delta) ** 2)

        C = np.zeros(self.config.NXK)
        C[0] = self.config.DTK * v * math.sin(phi) * phi
        C[1] = -self.config.DTK * v * math.cos(phi) * phi
        C[3] = -self.config.DTK * v * delta / (self.config.WB * math.cos(delta) ** 2)
        return A, B, C

    def mpc_prob_solve(self, ref_traj, path_predict, x0):
        self.x0k.value = x0

        A_block = []
        B_block = []
        C_block = []
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], 0.0
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        self.Annz_k.value = A_block.data
        self.Bnnz_k.value = B_block.data
        self.Ck_.value = C_block

        self.ref_traj_k.value = ref_traj

        self.MPC_prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)

        if (
            self.MPC_prob.status == cvxpy.OPTIMAL
            or self.MPC_prob.status == cvxpy.OPTIMAL_INACCURATE
        ):
            ox = np.array(self.xk.value[0, :]).flatten()
            oy = np.array(self.xk.value[1, :]).flatten()
            ov = np.array(self.xk.value[2, :]).flatten()
            oyaw = np.array(self.xk.value[3, :]).flatten()
            oa = np.array(self.uk.value[0, :]).flatten()
            odelta = np.array(self.uk.value[1, :]).flatten()
        else:
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None
        return oa, odelta, ox, oy, oyaw, ov

    def linear_mpc_control(self, ref_path, x0, oa, od):
        if oa is None or od is None:
            oa = [0.0] * self.config.TK
            od = [0.0] * self.config.TK

        path_predict = self.predict_motion(x0, oa, od, ref_path)
        poa, pod = oa[:], od[:]
        mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = self.mpc_prob_solve(
            ref_path, path_predict, x0
        )
        return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v, path_predict

    def update_plot(self):
        if self.points:
            points_array = np.array(self.points)
            self.trajectory_plot.setData(points_array[:, 0], points_array[:, 1])
        if self.current_point:
            self.current_location_plot.setData([{'pos': self.current_point[0], 'data': 1}])


def main(args=None):
    rclpy.init(args=args)
    mpc_node = MPC()

    executor_thread = threading.Thread(target=rclpy.spin, args=(mpc_node,), daemon=True)
    executor_thread.start()

    QApplication.instance().exec_()

    mpc_node.destroy_node()
    rclpy.shutdown()
    executor_thread.join()


if __name__ == '__main__':
    main()
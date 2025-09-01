#!/usr/bin/env python3
# ROS 2 Humble - EKF (3D) for IMU + GNSS
# State: [x, y, z, vx, vy, vz, yaw]
# Publishes local ENU odometry (x, y, z)

import math
import numpy as np

if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, TransformStamped
from custom_interfaces.msg import GGA
import pymap3d as pm
import tf_transformations
from tf2_ros import TransformBroadcaster

# ---------- Coordinate helpers ----------
def llh_to_enu(lat, lon, h, lat0, lon0, h0):
    e, n, u = pm.geodetic2enu(lat, lon, h, lat0, lon0, h0)
    return e, n, u

def yaw_from_quaternion(q: Quaternion) -> float:
    _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
    return yaw

def quaternion_from_yaw(yaw: float) -> Quaternion:
    qx, qy, qz, qw = tf_transformations.quaternion_from_euler(0.0, 0.0, yaw)
    q = Quaternion()
    q.x, q.y, q.z, q.w = qx, qy, qz, qw
    return q

@dataclass
class EKFParams:
    q_pos: float = 0.05      # process noise on position (continuous -> discretised with dt)
    q_vel: float = 0.5       # process noise on velocity
    q_yaw: float = 0.05      # process noise on yaw
    r_gps_pos: float = 2.5   # GPS measurement variance (fallback)
    r_yaw_meas: float = 0.3  # default IMU orientation yaw variance (fallback)
    use_imu_orientation: bool = True  # enable yaw fusion
    assume_linear_accel_is_gravity_removed: bool = False  # set False because your IMU includes gravity
    gravity_m_s2: float = 9.80665
    topic_imu: str = '/imu/data'
    topic_gps: str = '/gnss_data'
    topic_odom: str = '/ekf/odom'
    frame_map: str = 'map'
    frame_base: str = 'base_link'

class EKFLocalisation(Node):
    def __init__(self):
        super().__init__('ekf_localisation')

        # Parameters
        self.params = EKFParams()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Sub/Pub
        self.sub_imu = self.create_subscription(Imu, self.params.topic_imu, self.imu_cb, qos)
        self.sub_gps = self.create_subscription(GGA, self.params.topic_gps, self.gps_cb, qos)
        self.pub_odom = self.create_publisher(Odometry, self.params.topic_odom, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # EKF state [x, y, z, vx, vy, vz, yaw]
        self.x = np.zeros((7, 1))
        self.P = np.diag([10.0, 10.0, 10.0, 5.0, 5.0, 5.0, (math.pi/2.0)])

        self.last_imu_time = None
        self.have_origin = False
        self.lat0 = self.lon0 = self.h0 = 0.0

        self.last_gz = 0.0
        self.get_logger().info("3D EKF localisation node started (gravity-removal + quaternion rotation enabled).")

    # ----------------- Callbacks -----------------
    def imu_cb(self, msg: Imu):
        # convert stamp to seconds
        t = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        if self.last_imu_time is None:
            self.last_imu_time = t
            # initialize yaw from IMU orientation if present
            if msg.orientation_covariance[0] >= 0.0:
                self.x[6,0] = yaw_from_quaternion(msg.orientation)
            return

        dt = max(1e-3, t - self.last_imu_time)
        self.last_imu_time = t

        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z
        gz = msg.angular_velocity.z
        self.last_gz = gz

        # By default we will try to use orientation to remove gravity and rotate
        accel_in_world = False
        ax_w = ay_w = az_w = 0.0

        if (not self.params.assume_linear_accel_is_gravity_removed) and (msg.orientation_covariance[0] >= 0.0):
            # Build rotation matrix from quaternion (body -> world)
            q = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
            R = tf_transformations.quaternion_matrix(q)[:3, :3]  # 3x3

            acc_body = np.array([ax, ay, az], dtype=float)   # includes gravity
            # Rotate into world
            acc_world_raw = R.dot(acc_body)
            gvec = np.array([0.0, 0.0, self.params.gravity_m_s2])
            acc_world = acc_world_raw - gvec

            ax_w, ay_w, az_w = float(acc_world[0]), float(acc_world[1]), float(acc_world[2])
            accel_in_world = True
        else:
            # fallback: orientation not available -> we'll use yaw-only rotation inside predict
            ax_w, ay_w, az_w = ax, ay, az
            accel_in_world = False

        # Prediction step: note we pass either world-frame accel (gravity removed) or body-frame accel with flag
        self.ekf_predict(dt, ax_w, ay_w, az_w, gz, accel_in_world=accel_in_world)

        # Optional yaw measurement update using IMU quaternion as observation
        if self.params.use_imu_orientation and msg.orientation_covariance[0] >= 0.0:
            meas_yaw = yaw_from_quaternion(msg.orientation)
            # Use IMU-provided orientation covariance as a heuristic for yaw variance (if positive), else fallback param
            cov0 = float(msg.orientation_covariance[0])
            if cov0 > 0.0:
                r_yaw = cov0
            else:
                r_yaw = self.params.r_yaw_meas
            self.ekf_yaw_update(meas_yaw, r_yaw)

        # Publish odom with IMU timestamp for smoothness
        self.publish_odom(msg.header.stamp)

    def gps_cb(self, msg: GGA):
        # Set origin on first fix
        if not self.have_origin:
            self.lat0, self.lon0, self.h0 = msg.latitude, msg.longitude, msg.altitude
            self.have_origin = True
            self.get_logger().info(f"Set ENU origin at lat={self.lat0:.8f}, lon={self.lon0:.8f}, h={self.h0:.2f} m")

        # Convert to ENU (easting, northing, up)
        e, n, u = llh_to_enu(msg.latitude, msg.longitude, msg.altitude,
                             self.lat0, self.lon0, self.h0)
        z = np.array([[e], [n], [u]])
        # Use fixed R unless your GGA message provides per-fix covariance (then use it)
        R = np.eye(3) * self.params.r_gps_pos
        self.ekf_gps_update(z, R)
        # Publish odom with ROS time for GNSS updates
        self.publish_odom(self.get_clock().now().to_msg())

    # ----------------- EKF -----------------
    def ekf_predict(self, dt: float, ax: float, ay: float, az: float, gz: float, accel_in_world: bool = False):
        """
        State: [x, y, z, vx, vy, vz, yaw]
        If accel_in_world == True, ax/ay/az are expected to be in world frame (gravity removed).
        If accel_in_world == False, ax/ay/az are body-frame and will be rotated using yaw-only fallback.
        """
        x, y, z, vx, vy, vz, yaw = self.x.flatten()

        if accel_in_world:
            ax_world = ax
            ay_world = ay
            az_world = az
        else:
            # fallback: rotate using yaw only (cheap approximation)
            cy = math.cos(yaw)
            sy = math.sin(yaw)
            ax_world = cy*ax - sy*ay
            ay_world = sy*ax + cy*ay
            az_world = az  # still un-compensated in fallback

        # Discrete motion model
        x_pred  = x + vx*dt + 0.5*ax_world*dt*dt
        y_pred  = y + vy*dt + 0.5*ay_world*dt*dt
        z_pred  = z + vz*dt + 0.5*az_world*dt*dt
        vx_pred = vx + ax_world*dt
        vy_pred = vy + ay_world*dt
        vz_pred = vz + az_world*dt
        yaw_pred = self.wrap_angle(yaw + gz*dt)

        self.x = np.array([[x_pred],[y_pred],[z_pred],[vx_pred],[vy_pred],[vz_pred],[yaw_pred]])

        # Build Jacobian F (discrete). We use a simple block form:
        F = np.eye(7)
        F[0,3] = dt
        F[1,4] = dt
        F[2,5] = dt

        # If we used yaw-only rotation (fallback), we can include yaw partials analytically.
        # If we used full quaternion rotation (accel_in_world=True), the accel -> yaw partials
        # are more complex; for simplicity we leave them zero (pragmatic approximation).
        if not accel_in_world:
            # partial derivatives of rotated accel w.r.t yaw (body-frame inputs ax,ay)
            # indices: vx=3, vy=4, yaw=6
            cy = math.cos(yaw)
            sy = math.sin(yaw)
            # these come from ∂(0.5*dt^2 * rotated_accel)/∂yaw for position and dt*... for velocity
            F[0,6] = 0.5 * dt*dt * (-sy*ax - cy*ay)
            F[1,6] = 0.5 * dt*dt * ( cy*ax - sy*ay)
            F[3,6] = dt * (-sy*ax - cy*ay)
            F[4,6] = dt * ( cy*ax - sy*ay)
        # else: keep the yaw partials at zero (approx)

        # Process noise (discrete approx)
        q = self.params
        Q = np.diag([q.q_pos*dt]*3 + [q.q_vel*dt]*3 + [q.q_yaw*dt])
        self.P = F @ self.P @ F.T + Q

    def ekf_gps_update(self, z: np.ndarray, R: np.ndarray):
        # Measurement model: position only [x, y, z]
        H = np.zeros((3,7))
        H[0,0] = 1.0
        H[1,1] = 1.0
        H[2,2] = 1.0

        y = z - H @ self.x                        # innovation
        S = H @ self.P @ H.T + R                  # innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)       # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ H) @ self.P

        # Normalize yaw
        self.x[6,0] = self.wrap_angle(self.x[6,0])

    def ekf_yaw_update(self, yaw_meas: float, r_var: float):
        H = np.zeros((1,7))
        H[0,6] = 1.0
        innov = self.wrap_angle(yaw_meas - self.x[6,0])
        S = H @ self.P @ H.T + np.array([[r_var]])
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ np.array([[innov]])
        self.P = (np.eye(7) - K @ H) @ self.P
        self.x[6,0] = self.wrap_angle(self.x[6,0])

    @staticmethod
    def wrap_angle(a: float) -> float:
        return (a + math.pi) % (2.0*math.pi) - math.pi

    # ----------------- Publish -----------------
    def publish_odom(self, stamp):
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.params.frame_map
        odom.child_frame_id = self.params.frame_base

        x, y, z, vx, vy, vz, yaw = self.x.flatten()
        odom.pose.pose.position.x = float(x)
        odom.pose.pose.position.y = float(y)
        odom.pose.pose.position.z = float(z)
        odom.pose.pose.orientation = quaternion_from_yaw(float(yaw))

        # Fill pose covariance (6x6 flattened) with our uncertainties (partial mapping)
        pose_cov = np.zeros((6,6))
        pose_cov[0,0] = self.P[0,0]
        pose_cov[1,1] = self.P[1,1]
        pose_cov[2,2] = self.P[2,2]
        pose_cov[5,5] = self.P[6,6]  # yaw variance in the (5,5) slot
        odom.pose.covariance = pose_cov.flatten().tolist()

        odom.twist.twist.linear.x = float(vx)
        odom.twist.twist.linear.y = float(vy)
        odom.twist.twist.linear.z = float(vz)
        odom.twist.twist.angular.z = float(self.last_gz)

        twist_cov = np.zeros((6,6))
        twist_cov[0,0] = self.P[3,3]
        twist_cov[1,1] = self.P[4,4]
        twist_cov[2,2] = self.P[5,5]
        twist_cov[5,5] = max(self.P[6,6], 1e-6)
        odom.twist.covariance = twist_cov.flatten().tolist()

        self.pub_odom.publish(odom)

        if self.params.frame_map and self.params.frame_base:
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self.params.frame_map
            tf.child_frame_id = self.params.frame_base
            tf.transform.translation.x = float(x)
            tf.transform.translation.y = float(y)
            tf.transform.translation.z = float(z)
            tf.transform.rotation = odom.pose.pose.orientation
            self.tf_broadcaster.sendTransform(tf)

def main():
    rclpy.init()
    node = EKFLocalisation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
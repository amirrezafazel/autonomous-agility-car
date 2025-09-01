#!/usr/bin/env python3
# ROS2 Humble EKF (IMU + GNSS) localisation:
#  - assumes incoming Imu.linear_acceleration is gravity-remove
#  - removes redundant gravity-compensation branches
#  - clamps dt to avoid integration explosions on bad timestamps
# State vector:
#  [x, y, z, vx, vy, vz, yaw, bax, bay, baz, bgz]
import math
import numpy as np

if not hasattr(np, "float"):
    np.float = float

from dataclasses import dataclass
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu, MagneticField, NavSatFix, NavSatStatus
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, TransformStamped
from tf2_ros import TransformBroadcaster
import tf_transformations
import pymap3d as pm
# from custom_interfaces.msg import GGA

# Helpers functions:
# Normalize angle to [-pi, pi).
def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

# Extract yaw from geometry_msgs/Quaternion.
def yaw_from_quaternion_msg(q: Quaternion) -> float:
    return float(wrap_angle(tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]))

# Build quaternion for yaw only (roll=pitch=0).
def quaternion_from_yaw(yaw: float) -> Quaternion:
    qx, qy, qz, qw = tf_transformations.quaternion_from_euler(0.0, 0.0, float(yaw))
    q = Quaternion()
    q.x, q.y, q.z, q.w = float(qx), float(qy), float(qz), float(qw)
    return q

# Convert lat/lon/height to ENU (global to local coordinates).
def llh_to_enu(lat, lon, h, lat0, lon0, h0):
    e, n, u = pm.geodetic2enu(lat, lon, h, lat0, lon0, h0)
    return float(e), float(n), float(u)

# Compute heading from magnetometer.
def mag_heading_from_magmsg(mag_msg: MagneticField, orientation_q: Quaternion = None) -> float:
    mx = mag_msg.magnetic_field.x
    my = mag_msg.magnetic_field.y
    mz = mag_msg.magnetic_field.z
    # If orientation provided, rotate mag to world frame.
    if orientation_q is not None:
        q = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        R = tf_transformations.quaternion_matrix(q)[:3, :3]
        mag_body = np.array([mx, my, mz], dtype=float)
        mag_world = R.dot(mag_body)
        mx_w, my_w = float(mag_world[0]), float(mag_world[1])
        yaw = math.atan2(my_w, mx_w)
    # Use body frame x/y directly if no orientation.
    else:
        yaw = math.atan2(my, mx)
    return float(wrap_angle(yaw))

# Parameters
@dataclass
class EKFParams:
    # Process/measurement noise and configuration defaults.
    q_acc: float = 0.5
    q_bias_acc: float = 1e-4
    q_bias_gz: float = 1e-6
    q_gyro: float = 1e-3
    r_gps_pos: float = 4.0
    r_yaw_imu: float = 0.5
    r_yaw_mag: float = 0.8
    use_imu_orientation: bool = True
    use_mag: bool = True
    linear_accel_is_gravity_removed: bool = True
    gravity_m_s2: float = 9.80665
    topic_imu: str = '/ekf/imu/data'
    topic_mag: str = '/imu/mag'
    topic_gps: str = '/gnss/data'
    topic_odom: str = '/ekf/odom'
    frame_map: str = 'map'
    frame_base: str = 'base_link'
    regularisation_eps: float = 1e-8
    # Safety dt limits (seconds)
    dt_max: float = 0.2
    dt_min: float = 1e-4

class EKFLocalisation(Node):
    def __init__(self):
        super().__init__('ekf_localisation_biases')
        self.params = EKFParams()
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)

        self.sub_imu = self.create_subscription(Imu, self.params.topic_imu, self.imu_cb, qos)
        self.sub_mag = self.create_subscription(MagneticField, self.params.topic_mag, self.mag_cb, qos)
        self.sub_gps = self.create_subscription(NavSatFix, self.params.topic_gps, self.gps_cb, qos)

        self.pub_odom = self.create_publisher(Odometry, self.params.topic_odom, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # State size and state vector initialization.
        self.nx = 11
        self.x = np.zeros((self.nx, 1), dtype=float)

        # Initial covariance (diagonal)
        pos_var = 25.0
        vel_var = 4.0
        yaw_var = (math.pi/2.0)**2
        bias_acc_var = 0.5
        bias_gz_var = 1e-3
        self.P = np.diag([pos_var, pos_var, pos_var,
                          vel_var, vel_var, vel_var,
                          yaw_var,
                          bias_acc_var, bias_acc_var, bias_acc_var,
                          bias_gz_var])

        self.last_imu_time = None
        self.have_origin = False
        self.lat0 = self.lon0 = self.h0 = 0.0
        self.last_mag_msg = None
        self.last_gz = 0.0

        self.get_logger().info("EKF (gravity-removed accel) started.")

    def imu_cb(self, msg: Imu):
        # Build timestamp in seconds (float).
        t = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

        # First IMU message - initialize time and yaw.
        if self.last_imu_time is None:
            self.last_imu_time = t
            orient_ok = (len(msg.orientation_covariance) > 0 and msg.orientation_covariance[0] >= 0.0)
            if orient_ok and self.params.use_imu_orientation:
                self.x[6, 0] = yaw_from_quaternion_msg(msg.orientation)
                self.get_logger().info(f"Initial yaw set from IMU orientation: {float(self.x[6,0]):.3f} rad")
            return

        dt = t - self.last_imu_time

        # dt checks to avoid integration explosions on bad timestamps.
        if dt <= 0.0:
            self.get_logger().warning(f"Non-positive dt from IMU timestamps: dt={dt:.6f}. Ignoring and using dt_min.")
            dt = self.params.dt_min
        elif dt > self.params.dt_max:
            self.get_logger().warning(f"Large dt from IMU timestamps: dt={dt:.3f}s -> capped to {self.params.dt_max}s.")
            dt = self.params.dt_max

        self.last_imu_time = t

        # Read measurements.
        ax_m = float(msg.linear_acceleration.x)
        ay_m = float(msg.linear_acceleration.y)
        az_m = float(msg.linear_acceleration.z)
        gz_m = float(msg.angular_velocity.z)
        self.last_gz = gz_m

        # Biases from state (acceleration biases in body frame, gyro z bias scalar).
        bax = float(self.x[7, 0])
        bay = float(self.x[8, 0])
        baz = float(self.x[9, 0])
        bgz = float(self.x[10, 0])

        # Orientation availability check
        orient_ok = (len(msg.orientation_covariance) > 0 and msg.orientation_covariance[0] >= 0.0)

        # Subtract acceleration biases (body frame).
        acc_body = np.array([ax_m - bax, ay_m - bay, az_m - baz], dtype=float)

        R = None
        if orient_ok:
            # Rotate body frame acceleration into world frame.
            q = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
            R = tf_transformations.quaternion_matrix(q)[:3, :3]
            acc_world = R.dot(acc_body)
            ax_w, ay_w, az_w = float(acc_world[0]), float(acc_world[1]), float(acc_world[2])
            accel_in_world = True
        else:
            # Fallback: yaw-only rotate (approx).
            ax_b, ay_b, az_b = float(acc_body[0]), float(acc_body[1]), float(acc_body[2])
            ax_w, ay_w, az_w = ax_b, ay_b, az_b
            accel_in_world = False

        # gyro z after bias correction.
        gz_corr = gz_m - bgz

        # Small debug: if acceleration magnitude large while stationary, log it.
        acc_mag = math.sqrt(ax_w*ax_w + ay_w*ay_w + az_w*az_w)
        if acc_mag > 5.0:  # very large acceleration
            self.get_logger().warning(f"Large accel magnitude: {acc_mag:.3f} m/s^2 (ax,ay,az) = {ax_w:.3f},{ay_w:.3f},{az_w:.3f}")

        # Prediction step of EKF
        self.ekf_predict(dt, ax_w, ay_w, az_w, gz_corr, accel_in_world, R if accel_in_world and orient_ok else None)

        # Yaw update with IMU orientation (coarse)
        if self.params.use_imu_orientation and orient_ok:
            cov0 = float(msg.orientation_covariance[0])
            # If IMU provided covariance, use it, otherwise fall back to configured variance.
            r_yaw = cov0 if cov0 > 0.0 else self.params.r_yaw_imu
            meas_yaw = yaw_from_quaternion_msg(msg.orientation)
            self.ekf_yaw_update(meas_yaw, r_yaw)

        # Publish current odometry + TF.
        self.publish_odom(msg.header.stamp)

    def mag_cb(self, msg: MagneticField):
        self.last_mag_msg = msg
        if not self.params.use_mag:
            return
        try:
            yaw_meas = mag_heading_from_magmsg(msg, None)
            self.ekf_yaw_update(yaw_meas, self.params.r_yaw_mag)
        except Exception as e:
            self.get_logger().debug(f"mag_cb error: {e}")

    def gps_cb(self, msg: NavSatFix):
        lat = float(getattr(msg, 'latitude', 0.0))
        lon = float(getattr(msg, 'longitude', 0.0))
        alt = float(getattr(msg, 'altitude', 0.0) or 0.0)

        # Set first GPS fix as ENU origin.
        if not self.have_origin:
            self.lat0, self.lon0, self.h0 = lat, lon, alt
            self.have_origin = True
            self.get_logger().info(f"ENU origin set: lat={self.lat0:.8f}, lon={self.lon0:.8f}, h={self.h0:.2f}")
            self.x[0:3, 0] = 0.0

        try:
            e, n, u = llh_to_enu(lat, lon, alt, self.lat0, self.lon0, self.h0)
        except Exception as ex:
            self.get_logger().warn(f"ENU conversion failed: {ex}")
            return

        # Measurement vector and measurement covariance.
        z = np.array([[e], [n], [u]], dtype=float)
        R = np.eye(3) * float(self.params.r_gps_pos)
        R += np.eye(3) * self.params.regularisation_eps
        self.ekf_gps_update(z, R)

        # Update published odom after GPS update
        self.publish_odom(msg.header.stamp)

    def ekf_predict(self, dt: float, ax_w: float, ay_w: float, az_w: float, gz_corr: float, accel_in_world: bool, R_body_to_world=None):
        # Unpack state for clarity.
        x = self.x.flatten()
        px, py, pz, vx, vy, vz, yaw = x[0:7]
        bax, bay, baz, bgz = x[7], x[8], x[9], x[10]

        # If acceleration is body frame, rotate by yaw (approx).
        if not accel_in_world:
            cy = math.cos(yaw); sy = math.sin(yaw)
            ax_world = cy * ax_w - sy * ay_w
            ay_world = sy * ax_w + cy * ay_w
            az_world = az_w
            R_for_bias = np.array([[cy, -sy, 0.0],[sy, cy, 0.0],[0.0,0.0,1.0]], dtype=float)
        # If acceleration already in world frame (because IMU orientation known).
        else:
            ax_world, ay_world, az_world = ax_w, ay_w, az_w
            R_for_bias = R_body_to_world if R_body_to_world is not None else np.eye(3)

        # Constant acceleration motion model.
        px_pred = px + vx * dt + 0.5 * ax_world * dt * dt
        py_pred = py + vy * dt + 0.5 * ay_world * dt * dt
        pz_pred = pz + vz * dt + 0.5 * az_world * dt * dt
        vx_pred = vx + ax_world * dt
        vy_pred = vy + ay_world * dt
        vz_pred = vz + az_world * dt
        yaw_pred = wrap_angle(yaw + gz_corr * dt)

        # Biases modeled as random walk (remain the same in prediction).
        bax_pred, bay_pred, baz_pred, bgz_pred = bax, bay, baz, bgz

        # Write predicted state back.
        self.x = np.array([[px_pred], [py_pred], [pz_pred],
                           [vx_pred], [vy_pred], [vz_pred],
                           [yaw_pred],
                           [bax_pred], [bay_pred], [baz_pred],
                           [bgz_pred]], dtype=float)

        # Build state transition Jacobian F.
        F = np.eye(self.nx, dtype=float)
        F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt

        # Coupling of acceleration biases into velocity and position.
        F[3, 7] = -dt * float(R_for_bias[0, 0])
        F[3, 8] = -dt * float(R_for_bias[0, 1])
        F[3, 9] = -dt * float(R_for_bias[0, 2])
        F[4, 7] = -dt * float(R_for_bias[1, 0])
        F[4, 8] = -dt * float(R_for_bias[1, 1])
        F[4, 9] = -dt * float(R_for_bias[1, 2])
        F[5, 7] = -dt * float(R_for_bias[2, 0])
        F[5, 8] = -dt * float(R_for_bias[2, 1])
        F[5, 9] = -dt * float(R_for_bias[2, 2])

        F[0, 7] = -0.5 * dt * dt * float(R_for_bias[0, 0])
        F[0, 8] = -0.5 * dt * dt * float(R_for_bias[0, 1])
        F[0, 9] = -0.5 * dt * dt * float(R_for_bias[0, 2])
        F[1, 7] = -0.5 * dt * dt * float(R_for_bias[1, 0])
        F[1, 8] = -0.5 * dt * dt * float(R_for_bias[1, 1])
        F[1, 9] = -0.5 * dt * dt * float(R_for_bias[1, 2])
        F[2, 7] = -0.5 * dt * dt * float(R_for_bias[2, 0])
        F[2, 8] = -0.5 * dt * dt * float(R_for_bias[2, 1])
        F[2, 9] = -0.5 * dt * dt * float(R_for_bias[2, 2])

        # Yaw depends on gyro bias.
        F[6, 10] = -dt

        # Process noise covariance Q (simple diagonal model).
        q = self.params
        q_pos = q.q_acc * (dt ** 2)
        q_vel = q.q_acc * dt
        q_yaw = q.q_gyro * dt
        Q = np.zeros((self.nx, self.nx), dtype=float)
        Q[0, 0] = q_pos; Q[1, 1] = q_pos; Q[2, 2] = q_pos
        Q[3, 3] = q_vel; Q[4, 4] = q_vel; Q[5, 5] = q_vel
        Q[6, 6] = q_yaw
        Q[7, 7] = q.q_bias_acc * dt
        Q[8, 8] = q.q_bias_acc * dt
        Q[9, 9] = q.q_bias_acc * dt
        Q[10, 10] = q.q_bias_gz * dt

        # Covariance prediction
        self.P = F @ self.P @ F.T + Q

    def ekf_gps_update(self, z: np.ndarray, R: np.ndarray):
        # Build measurement matrix H to extract x,y,z from the state vector.
        H = np.zeros((3, self.nx), dtype=float)
        H[0, 0] = 1.0; H[1, 1] = 1.0; H[2, 2] = 1.0

        # Innovation - difference between GPS and prediction.
        y = z - (H @ self.x)
        # Innovation covariance
        S = H @ self.P @ H.T + R
        S += np.eye(3) * self.params.regularisation_eps
        try:
            # Kalman gain - how much to trust GPS vs prediction
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.get_logger().warn("GPS update failed (singular S).")
            return
        # Apply Kalman gain.
        self.x = self.x + K @ y
        # Covariance update
        self.P = (np.eye(self.nx) - K @ H) @ self.P
        # Normalize yaw after update.
        self.x[6, 0] = wrap_angle(self.x[6, 0])

    def ekf_yaw_update(self, yaw_meas: float, r_var: float):
        # Build measurement matrix H to extract yaw state.
        H = np.zeros((1, self.nx), dtype=float)
        H[0, 6] = 1.0
        # Innovation - difference between measured and predicted yaw (normalised).
        innov = wrap_angle(float(yaw_meas) - float(self.x[6, 0]))
        # Innovation covariance (with small regularization for stability).
        S = H @ self.P @ H.T + np.array([[r_var + self.params.regularisation_eps]], dtype=float)
        try:
            # Kalman gain
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.get_logger().warn("Yaw update failed (singular S).")
            return
        # Apply Kalman gain.
        self.x = self.x + K @ np.array([[innov]], dtype=float)
        # Covariance update
        self.P = (np.eye(self.nx) - K @ H) @ self.P
        # Normalize yaw after update.
        self.x[6, 0] = wrap_angle(self.x[6, 0])

    def publish_odom(self, stamp):
        # Publish Odometry message and TF between map and base_link.
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.params.frame_map
        odom.child_frame_id = self.params.frame_base

        # Unpack pose + velocity for message population.
        px, py, pz, vx, vy, vz, yaw = [float(v) for v in self.x[0:7, 0]]
        odom.pose.pose.position.x = px
        odom.pose.pose.position.y = py
        odom.pose.pose.position.z = pz
        odom.pose.pose.orientation = quaternion_from_yaw(yaw)

        # Pose covariance (6x6) - mapping from state covariances.
        pose_cov = np.zeros((6, 6), dtype=float)
        pose_cov[0, 0] = float(self.P[0, 0]); pose_cov[1, 1] = float(self.P[1, 1]); pose_cov[2, 2] = float(self.P[2, 2])
        pose_cov[5, 5] = float(self.P[6, 6])
        odom.pose.covariance = pose_cov.flatten().tolist()

        # twist (velocity) and twist covariance.
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.linear.z = vz
        odom.twist.twist.angular.z = float(self.last_gz)

        twist_cov = np.zeros((6, 6), dtype=float)
        twist_cov[0, 0] = float(self.P[3, 3]); twist_cov[1, 1] = float(self.P[4, 4]); twist_cov[2, 2] = float(self.P[5, 5])
        twist_cov[5, 5] = max(float(self.P[6, 6]), 1e-12)
        odom.twist.covariance = twist_cov.flatten().tolist()

        # Publish odometry.
        self.pub_odom.publish(odom)

        # Broadcast TF.
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self.params.frame_map
        tf.child_frame_id = self.params.frame_base
        tf.transform.translation.x = px
        tf.transform.translation.y = py
        tf.transform.translation.z = pz
        tf.transform.rotation = odom.pose.pose.orientation
        self.tf_broadcaster.sendTransform(tf)

def main(args=None):
    rclpy.init(args=args)
    node = EKFLocalisation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
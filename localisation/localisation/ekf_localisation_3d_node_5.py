#!/usr/bin/env python3
"""
ROS2 Humble EKF (IMU + GNSS) simplified:
 - assumes incoming Imu.linear_acceleration is gravity-removed
 - removes redundant gravity-compensation branches
 - clamps dt to avoid integration explosions on bad timestamps
State vector (11):
 [x, y, z, vx, vy, vz, yaw, bax, bay, baz, bgz]
"""
import math
import numpy as np

if not hasattr(np, "float"):
    np.float = float

from dataclasses import dataclass
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu, MagneticField
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, TransformStamped
from tf2_ros import TransformBroadcaster
import tf_transformations
import pymap3d as pm

from custom_interfaces.msg import GGA

# helpers
def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def yaw_from_quaternion_msg(q: Quaternion) -> float:
    return float(wrap_angle(tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]))

def quaternion_from_yaw(yaw: float) -> Quaternion:
    qx, qy, qz, qw = tf_transformations.quaternion_from_euler(0.0, 0.0, float(yaw))
    q = Quaternion()
    q.x, q.y, q.z, q.w = float(qx), float(qy), float(qz), float(qw)
    return q

def llh_to_enu(lat, lon, h, lat0, lon0, h0):
    e, n, u = pm.geodetic2enu(lat, lon, h, lat0, lon0, h0)
    return float(e), float(n), float(u)

def nmea_latlon_to_decimal(value: float, hemi: str, is_lat: bool) -> float:
    if value is None:
        return value
    if (is_lat and abs(value) > 90.0) or (not is_lat and abs(value) > 180.0):
        deg = int(value // 100)
        minutes = value - deg * 100
        dec = deg + minutes / 60.0
    else:
        dec = value
    if hemi in ('S', 's', 'W', 'w'):
        dec = -abs(dec)
    return float(dec)

def mag_heading_from_magmsg(mag_msg: MagneticField, orientation_q: Quaternion = None) -> float:
    mx = mag_msg.magnetic_field.x
    my = mag_msg.magnetic_field.y
    mz = mag_msg.magnetic_field.z
    if orientation_q is not None:
        q = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        R = tf_transformations.quaternion_matrix(q)[:3, :3]
        mag_body = np.array([mx, my, mz], dtype=float)
        mag_world = R.dot(mag_body)
        mx_w, my_w = float(mag_world[0]), float(mag_world[1])
        yaw = math.atan2(my_w, mx_w)
    else:
        yaw = math.atan2(my, mx)
    return float(wrap_angle(yaw))

# params
@dataclass
class EKFParams:
    q_acc: float = 0.5
    q_bias_acc: float = 1e-4
    q_bias_gz: float = 1e-6
    q_gyro: float = 1e-3
    r_gps_pos: float = 4.0
    r_yaw_imu: float = 0.5
    r_yaw_mag: float = 0.8
    use_imu_orientation: bool = True
    use_mag: bool = True
    # IMPORTANT: expect IMU.linear_acceleration to be gravity removed
    assume_linear_accel_is_gravity_removed: bool = True
    gravity_m_s2: float = 9.80665
    topic_imu: str = '/ekf/imu/data'
    topic_mag: str = '/imu/mag'
    topic_gps: str = '/gnss_data'
    topic_odom: str = '/ekf/odom'
    frame_map: str = 'map'
    frame_base: str = 'base_link'
    regularisation_eps: float = 1e-8
    # safety dt limits (seconds)
    dt_max: float = 0.2
    dt_min: float = 1e-4

class EKFLocalisation(Node):
    def __init__(self):
        super().__init__('ekf_localisation_biases')
        self.params = EKFParams()
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=10)

        self.sub_imu = self.create_subscription(Imu, self.params.topic_imu, self.imu_cb, qos)
        self.sub_mag = self.create_subscription(MagneticField, self.params.topic_mag, self.mag_cb, qos)
        self.sub_gps = self.create_subscription(GGA, self.params.topic_gps, self.gps_cb, qos)

        self.pub_odom = self.create_publisher(Odometry, self.params.topic_odom, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.nx = 11
        self.x = np.zeros((self.nx, 1), dtype=float)

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

        self.get_logger().info("EKF (gravity-removed accel assumed) started.")

    def imu_cb(self, msg: Imu):
        # build timestamp
        t = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

        if self.last_imu_time is None:
            self.last_imu_time = t
            orient_ok = (len(msg.orientation_covariance) > 0 and msg.orientation_covariance[0] >= 0.0)
            if orient_ok and self.params.use_imu_orientation:
                self.x[6, 0] = yaw_from_quaternion_msg(msg.orientation)
                self.get_logger().info(f"Initial yaw set from IMU orientation: {float(self.x[6,0]):.3f} rad")
            return

        dt = t - self.last_imu_time

        # defensive dt checks to avoid integration explosions
        if dt <= 0.0:
            self.get_logger().warning(f"Non-positive dt from IMU timestamps: dt={dt:.6f}. Ignoring and using dt_min.")
            dt = self.params.dt_min
        elif dt > self.params.dt_max:
            self.get_logger().warning(f"Large dt from IMU timestamps: dt={dt:.3f}s -> capped to {self.params.dt_max}s.")
            dt = self.params.dt_max

        self.last_imu_time = t

        # read measurements
        ax_m = float(msg.linear_acceleration.x)
        ay_m = float(msg.linear_acceleration.y)
        az_m = float(msg.linear_acceleration.z)
        gz_m = float(msg.angular_velocity.z)
        self.last_gz = gz_m

        # biases from state (accel biases in body frame, gyro z bias scalar)
        bax = float(self.x[7, 0])
        bay = float(self.x[8, 0])
        baz = float(self.x[9, 0])
        bgz = float(self.x[10, 0])

        # orientation availability
        orient_ok = (len(msg.orientation_covariance) > 0 and msg.orientation_covariance[0] >= 0.0)

        # assume IMU already removed gravity: subtract biases (body frame)
        acc_body = np.array([ax_m - bax, ay_m - bay, az_m - baz], dtype=float)

        R = None
        if orient_ok:
            q = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
            R = tf_transformations.quaternion_matrix(q)[:3, :3]
            acc_world = R.dot(acc_body)
            ax_w, ay_w, az_w = float(acc_world[0]), float(acc_world[1]), float(acc_world[2])
            accel_in_world = True
        else:
            # fallback: yaw-only rotate (approx)
            ax_b, ay_b, az_b = float(acc_body[0]), float(acc_body[1]), float(acc_body[2])
            ax_w, ay_w, az_w = ax_b, ay_b, az_b
            accel_in_world = False

        gz_corr = gz_m - bgz

        # small debug: if accel magnitude large while stationary, log it
        acc_mag = math.sqrt(ax_w*ax_w + ay_w*ay_w + az_w*az_w)
        if acc_mag > 5.0:  # very large acceleration
            self.get_logger().warning(f"Large accel magnitude: {acc_mag:.3f} m/s^2 (ax,ay,az) = {ax_w:.3f},{ay_w:.3f},{az_w:.3f}")

        # predict
        self.ekf_predict(dt, ax_w, ay_w, az_w, gz_corr, accel_in_world, R if accel_in_world and orient_ok else None)

        # yaw update with IMU orientation (coarse)
        if self.params.use_imu_orientation and orient_ok:
            cov0 = float(msg.orientation_covariance[0])
            r_yaw = cov0 if cov0 > 0.0 else self.params.r_yaw_imu
            meas_yaw = yaw_from_quaternion_msg(msg.orientation)
            self.ekf_yaw_update(meas_yaw, r_yaw)

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

    def gps_cb(self, msg: GGA):
        # ignore invalid fixes
        if msg.latitude == 0.0 and msg.longitude == 0.0:
            self.get_logger().info("Ignoring zero GNSS fix.")
            return

        lat = nmea_latlon_to_decimal(msg.latitude, getattr(msg, 'lat_dir', ''), is_lat=True)
        lon = nmea_latlon_to_decimal(msg.longitude, getattr(msg, 'lon_dir', ''), is_lat=False)
        alt = float(getattr(msg, 'altitude', 0.0) or 0.0)

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

        z = np.array([[e], [n], [u]], dtype=float)
        R = np.eye(3) * float(self.params.r_gps_pos)
        R += np.eye(3) * self.params.regularisation_eps
        self.ekf_gps_update(z, R)

        self.publish_odom(self.get_clock().now().to_msg())

    def ekf_predict(self, dt: float, ax_w: float, ay_w: float, az_w: float, gz_corr: float, accel_in_world: bool, R_body_to_world=None):
        x = self.x.flatten()
        px, py, pz, vx, vy, vz, yaw = x[0:7]
        bax, bay, baz, bgz = x[7], x[8], x[9], x[10]

        if not accel_in_world:
            cy = math.cos(yaw); sy = math.sin(yaw)
            ax_world = cy * ax_w - sy * ay_w
            ay_world = sy * ax_w + cy * ay_w
            az_world = az_w
            R_for_bias = np.array([[cy, -sy, 0.0],[sy, cy, 0.0],[0.0,0.0,1.0]], dtype=float)
        else:
            ax_world, ay_world, az_world = ax_w, ay_w, az_w
            R_for_bias = R_body_to_world if R_body_to_world is not None else np.eye(3)

        px_pred = px + vx * dt + 0.5 * ax_world * dt * dt
        py_pred = py + vy * dt + 0.5 * ay_world * dt * dt
        pz_pred = pz + vz * dt + 0.5 * az_world * dt * dt
        vx_pred = vx + ax_world * dt
        vy_pred = vy + ay_world * dt
        vz_pred = vz + az_world * dt
        yaw_pred = wrap_angle(yaw + gz_corr * dt)

        bax_pred, bay_pred, baz_pred, bgz_pred = bax, bay, baz, bgz

        self.x = np.array([[px_pred], [py_pred], [pz_pred],
                           [vx_pred], [vy_pred], [vz_pred],
                           [yaw_pred],
                           [bax_pred], [bay_pred], [baz_pred],
                           [bgz_pred]], dtype=float)

        F = np.eye(self.nx, dtype=float)
        F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt

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

        F[6, 10] = -dt

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

        self.P = F @ self.P @ F.T + Q

    def ekf_gps_update(self, z: np.ndarray, R: np.ndarray):
        H = np.zeros((3, self.nx), dtype=float)
        H[0, 0] = 1.0; H[1, 1] = 1.0; H[2, 2] = 1.0

        y = z - (H @ self.x)
        S = H @ self.P @ H.T + R
        S += np.eye(3) * self.params.regularisation_eps
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.get_logger().warn("GPS update failed (singular S).")
            return
        self.x = self.x + K @ y
        self.P = (np.eye(self.nx) - K @ H) @ self.P
        self.x[6, 0] = wrap_angle(self.x[6, 0])

    def ekf_yaw_update(self, yaw_meas: float, r_var: float):
        H = np.zeros((1, self.nx), dtype=float)
        H[0, 6] = 1.0
        innov = wrap_angle(float(yaw_meas) - float(self.x[6, 0]))
        S = H @ self.P @ H.T + np.array([[r_var + self.params.regularisation_eps]], dtype=float)
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.get_logger().warn("Yaw update failed (singular S).")
            return
        self.x = self.x + K @ np.array([[innov]], dtype=float)
        self.P = (np.eye(self.nx) - K @ H) @ self.P
        self.x[6, 0] = wrap_angle(self.x[6, 0])

    def publish_odom(self, stamp):
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.params.frame_map
        odom.child_frame_id = self.params.frame_base

        px, py, pz, vx, vy, vz, yaw = [float(v) for v in self.x[0:7, 0]]
        odom.pose.pose.position.x = px
        odom.pose.pose.position.y = py
        odom.pose.pose.position.z = pz
        odom.pose.pose.orientation = quaternion_from_yaw(yaw)

        pose_cov = np.zeros((6, 6), dtype=float)
        pose_cov[0, 0] = float(self.P[0, 0]); pose_cov[1, 1] = float(self.P[1, 1]); pose_cov[2, 2] = float(self.P[2, 2])
        pose_cov[5, 5] = float(self.P[6, 6])
        odom.pose.covariance = pose_cov.flatten().tolist()

        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.linear.z = vz
        odom.twist.twist.angular.z = float(self.last_gz)

        twist_cov = np.zeros((6, 6), dtype=float)
        twist_cov[0, 0] = float(self.P[3, 3]); twist_cov[1, 1] = float(self.P[4, 4]); twist_cov[2, 2] = float(self.P[5, 5])
        twist_cov[5, 5] = max(float(self.P[6, 6]), 1e-12)
        odom.twist.covariance = twist_cov.flatten().tolist()

        self.pub_odom.publish(odom)

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
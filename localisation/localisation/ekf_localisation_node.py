#!/usr/bin/env python3
# ROS 2 Humble - EKF (2D) for IMU + GNSS
# Fuses:
#   - IMU (/imu/data): linear accel (ax, ay), yaw-rate (gz), orientation (optional)
#   - GNSS (/gnss_data): GGA -> ENU position (x, y) update
#
# Publishes:
#   - /ekf/odom (nav_msgs/Odometry)
#   - TF map -> base_link (optional)
#
# Frames:
#   - map: local ENU tangent plane
#   - base_link: vehicle body frame (x forward, y left, z up per REP-103)

import math
import numpy as np
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, TransformStamped
from custom_interfaces.msg import GGA
import tf_transformations
from tf2_ros import TransformBroadcaster

# ---------- Optional ENU helpers ----------
def _try_import_geo():
    try:
        import pymap3d as pm
        return ('pymap3d', pm)
    except Exception:
        pass
    try:
        from geographiclib.geodesic import Geodesic
        return ('geographiclib', Geodesic.WGS84)
    except Exception:
        pass
    return (None, None)

_GEO_KIND, _GEO_LIB = _try_import_geo()

def llh_to_enu(lat, lon, h, lat0, lon0, h0):
    """
    Convert WGS84 (lat,lon,height) to local ENU relative to (lat0,lon0,h0).
    Uses pymap3d if available, else geographiclib, else small-angle fallback.
    Returns (e, n, u) in meters.
    """
    if _GEO_KIND == 'pymap3d':
        pm = _GEO_LIB
        e, n, u = pm.geodetic2enu(lat, lon, h, lat0, lon0, h0)
        return e, n, u
    elif _GEO_KIND == 'geographiclib':
        wgs84 = _GEO_LIB
        # Use forward-inverse on the ellipsoid for EN, and approximate Up by height diff.
        inv = wgs84.Inverse(lat0, lon0, lat, lon)
        s = inv['s12']          # surface distance
        azi_rad = math.radians(inv['azi1'])  # azimuth from ref to point
        n = s * math.cos(azi_rad)
        e = s * math.sin(azi_rad)
        u = (h - h0)
        return e, n, u
    else:
        # Small-area equirectangular fallback (ok for ~<10 km boxes)
        # WGS84 mean radii
        R_earth = 6378137.0
        dlat = math.radians(lat - lat0)
        dlon = math.radians(lon - lon0)
        lat_m = math.radians((lat + lat0) * 0.5)
        x = R_earth * dlon * math.cos(lat_m)
        y = R_earth * dlat
        z = (h - h0)
        return x, y, z

def yaw_from_quaternion(q: Quaternion) -> float:
    """Return yaw (rad) from geometry_msgs/Quaternion."""
    _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
    return yaw

def quaternion_from_yaw(yaw: float) -> Quaternion:
    qx, qy, qz, qw = tf_transformations.quaternion_from_euler(0.0, 0.0, yaw)
    q = Quaternion()
    q.x, q.y, q.z, q.w = qx, qy, qz, qw
    return q

@dataclass
class EKFParams:
    use_imu_orientation: bool = False  # if True, fuse IMU yaw as weak measurement
    publish_tf: bool = True

    # Process noise (continuous) — tune for your platform
    q_pos: float = 0.05     # m^2/s for process driving position via accel uncertainty
    q_vel: float = 0.5      # (m/s)^2/s
    q_yaw: float = 0.05     # rad^2/s (gyro bias/uncertainty proxy)

    # Measurement noise (GPS)
    r_gps_pos: float = 2.5  # m^2 (set ~ sigma^2 of your GNSS horizontal)
    # Optional yaw fusion from IMU orientation (weak)
    r_yaw_meas: float = 0.3 # rad^2

    # Gravity & accel usage
    assume_imu_linear_accel_is_gravity_removed: bool = True

    # Topic names & frames
    topic_imu: str = '/imu/data'
    topic_gps: str = '/gnss_data'
    topic_odom: str = '/ekf/odom'
    frame_map: str = 'map'
    frame_base: str = 'base_link'

class EKFLocalisation(Node):
    def __init__(self):
        super().__init__('ekf_localisation')

        # Load parameters (declare with defaults)
        self.params = EKFParams(
            use_imu_orientation = self.declare_parameter('use_imu_orientation', False).get_parameter_value().bool_value,
            publish_tf          = self.declare_parameter('publish_tf', True).get_parameter_value().bool_value,
            q_pos               = float(self.declare_parameter('q_pos', 0.05).value),
            q_vel               = float(self.declare_parameter('q_vel', 0.5).value),
            q_yaw               = float(self.declare_parameter('q_yaw', 0.05).value),
            r_gps_pos           = float(self.declare_parameter('r_gps_pos', 2.5).value),
            r_yaw_meas          = float(self.declare_parameter('r_yaw_meas', 0.3).value),
            assume_imu_linear_accel_is_gravity_removed = self.declare_parameter(
                'assume_imu_linear_accel_is_gravity_removed', True).get_parameter_value().bool_value,
            topic_imu           = self.declare_parameter('topic_imu', '/imu/data').get_parameter_value().string_value,
            topic_gps           = self.declare_parameter('topic_gps', '/gnss_data').get_parameter_value().string_value,
            topic_odom          = self.declare_parameter('topic_odom', '/ekf/odom').get_parameter_value().string_value,
            frame_map           = self.declare_parameter('frame_map', 'map').get_parameter_value().string_value,
            frame_base          = self.declare_parameter('frame_base', 'base_link').get_parameter_value().string_value,
        )

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.sub_imu = self.create_subscription(Imu, self.params.topic_imu, self.imu_cb, qos)
        self.sub_gps = self.create_subscription(GGA, self.params.topic_gps, self.gps_cb, qos)
        self.pub_odom = self.create_publisher(Odometry, self.params.topic_odom, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # EKF state: [x, y, vx, vy, yaw]
        self.x = np.zeros((5, 1))
        self.P = np.diag([10.0, 10.0, 5.0, 5.0, (math.pi/2.0)])  # fairly uninformative

        self.last_imu_time = None
        self.have_origin = False
        self.lat0 = self.lon0 = self.h0 = 0.0

        # Last IMU readings
        self.last_ax = 0.0
        self.last_ay = 0.0
        self.last_gz = 0.0
        self.last_imu_orientation = None

        self.get_logger().info("EKF localisation node started.")

    # ----------------- Callbacks -----------------
    def imu_cb(self, msg: Imu):
        t = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        if self.last_imu_time is None:
            self.last_imu_time = t
            # Initialize yaw from IMU orientation if present
            if msg.orientation_covariance[0] >= 0.0:
                self.x[4,0] = yaw_from_quaternion(msg.orientation)
                self.last_imu_orientation = msg.orientation
            return

        dt = max(1e-3, t - self.last_imu_time)  # guard
        self.last_imu_time = t

        # Extract body-frame measurements
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        gz = msg.angular_velocity.z

        # Save for reference
        self.last_ax, self.last_ay, self.last_gz = ax, ay, gz
        self.last_imu_orientation = msg.orientation

        # Prediction step (discrete-time approx)
        self.ekf_predict(dt, ax, ay, gz)

        # Optional weak yaw meas from orientation (helps drift if gyro biases)
        if self.params.use_imu_orientation and msg.orientation_covariance[0] >= 0.0:
            meas_yaw = yaw_from_quaternion(msg.orientation)
            self.ekf_yaw_update(meas_yaw, self.params.r_yaw_meas)

        # Publish odometry on every IMU update for smoothness
        self.publish_odom(msg.header.stamp)

    def gps_cb(self, msg: NavSatFix):
        if msg.status.status is None:
            pass  # tolerate
        if not self.have_origin:
            self.lat0, self.lon0, self.h0 = msg.latitude, msg.longitude, msg.altitude
            self.have_origin = True
            self.get_logger().info(f"Set ENU origin at lat={self.lat0:.8f}, lon={self.lon0:.8f}, h={self.h0:.2f} m")

        if msg.status.status is not None and msg.status.status < 0:
            # No valid fix; skip update
            return

        e, n, _ = llh_to_enu(msg.latitude, msg.longitude, msg.altitude,
                             self.lat0, self.lon0, self.h0)

        # Measurement: z = [x, y]
        z = np.array([[e], [n]])

        # Build R from reported covariance if available
        if len(msg.position_covariance) == 9 and msg.position_covariance[0] > 0.0:
            R = np.array([[msg.position_covariance[0], msg.position_covariance[1]],
                          [msg.position_covariance[3], msg.position_covariance[4]]])
        else:
            R = np.eye(2) * self.params.r_gps_pos

        self.ekf_gps_update(z, R)

        # Publish odometry with current ROS time if we don't have IMU stamps
        self.publish_odom(self.get_clock().now().to_msg())

    # ----------------- EKF math -----------------
    def ekf_predict(self, dt: float, ax_body: float, ay_body: float, gz: float):
        """
        State: [x, y, vx, vy, yaw]
        Inputs: body-frame linear accel (ax, ay), yaw rate gz (rad/s)
        """
        x, y, vx, vy, yaw = self.x.flatten()

        # Rotate accel to map frame (assuming map ~ ENU and base_link x-forward, y-left)
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        ax_world = cy * ax_body - sy * ay_body
        ay_world = sy * ax_body + cy * ay_body

        # If IMU linear acceleration still contains gravity, try to ignore z
        if not self.params.assume_imu_linear_accel_is_gravity_removed:
            # We don't know pitch/roll here; if your IMU gives good quaternion,
            # set `use_imu_orientation=true` and ensure linear_accel is gravity-removed.
            pass

        # Discrete motion model (CV + accel input)
        x_pred  = x + vx*dt + 0.5*ax_world*dt*dt
        y_pred  = y + vy*dt + 0.5*ay_world*dt*dt
        vx_pred = vx + ax_world*dt
        vy_pred = vy + ay_world*dt
        yaw_pred = self.wrap_angle(yaw + gz*dt)

        self.x = np.array([[x_pred], [y_pred], [vx_pred], [vy_pred], [yaw_pred]])

        # Jacobian F (df/dx) and process noise G Qc G^T
        F = np.eye(5)
        F[0,2] = dt
        F[1,3] = dt
        # partials wrt yaw (accel rotation)
        F[0,4] = 0.5 * dt*dt * (-sy*ax_body - cy*ay_body)
        F[1,4] = 0.5 * dt*dt * ( cy*ax_body - sy*ay_body)
        F[2,4] = dt * (-sy*ax_body - cy*ay_body)
        F[3,4] = dt * ( cy*ax_body - sy*ay_body)

        # Simple continuous process noise -> discrete (approx Qd = diag * dt)
        qx = self.params.q_pos
        qv = self.params.q_vel
        qy = self.params.q_yaw
        Q = np.diag([qx*dt, qx*dt, qv*dt, qv*dt, qy*dt])

        self.P = F @ self.P @ F.T + Q

    def ekf_gps_update(self, z: np.ndarray, R: np.ndarray):
        # Measurement model: H * x = [x, y]
        H = np.zeros((2,5))
        H[0,0] = 1.0
        H[1,1] = 1.0

        y = z - H @ self.x                        # innovation
        S = H @ self.P @ H.T + R                  # innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)       # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(5) - K @ H) @ self.P

        # Normalize yaw
        self.x[4,0] = self.wrap_angle(self.x[4,0])

    def ekf_yaw_update(self, yaw_meas: float, r_var: float):
        # H yaw picks the yaw element
        H = np.zeros((1,5))
        H[0,4] = 1.0
        # innovation with angle wrap
        innov = self.wrap_angle(yaw_meas - self.x[4,0])
        S = H @ self.P @ H.T + np.array([[r_var]])
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ np.array([[innov]])
        self.P = (np.eye(5) - K @ H) @ self.P
        self.x[4,0] = self.wrap_angle(self.x[4,0])

    @staticmethod
    def wrap_angle(a: float) -> float:
        return (a + math.pi) % (2.0*math.pi) - math.pi

    # ----------------- Publishing -----------------
    def publish_odom(self, stamp):
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.params.frame_map
        odom.child_frame_id = self.params.frame_base

        x, y, vx, vy, yaw = self.x.flatten()
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = 0.0

        odom.pose.pose.orientation = quaternion_from_yaw(yaw)

        # Fill pose covariance (flattened row-major 6x6; we place our 2D+heading in it)
        P = self.P
        pose_cov = np.zeros((6,6))
        pose_cov[0,0] = P[0,0]
        pose_cov[1,1] = P[1,1]
        pose_cov[5,5] = P[4,4]  # yaw in the (5,5) slot
        odom.pose.covariance = pose_cov.flatten().tolist()

        # Twist
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = self.last_gz

        twist_cov = np.zeros((6,6))
        twist_cov[0,0] = P[2,2]
        twist_cov[1,1] = P[3,3]
        twist_cov[5,5] = max(P[4,4], 1e-3)
        odom.twist.covariance = twist_cov.flatten().tolist()

        self.pub_odom.publish(odom)

        if self.params.publish_tf:
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self.params.frame_map
            tf.child_frame_id = self.params.frame_base
            tf.transform.translation.x = x
            tf.transform.translation.y = y
            tf.transform.translation.z = 0.0
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
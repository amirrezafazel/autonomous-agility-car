#!/usr/bin/env python3
# ROS 2 Humble - 3D EKF for IMU + GNSS (Hemisphere) + IMU pitch/roll (and optional yaw)
#
# Topics:
#   Sub: /imu/data   (sensor_msgs/Imu)
#   Sub: /fix        (sensor_msgs/NavSatFix)
#   Pub: /ekf/odom   (nav_msgs/Odometry)
#   TF:  map -> base_link (optional)
#
# Frames:
#   map: local ENU tangent plane
#   base_link: robot body, x-forward, y-left, z-up (REP-103)

import math
import numpy as np
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Imu, NavSatFix
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, TransformStamped
from tf2_ros import TransformBroadcaster
import tf_transformations

# ---------- ENU helpers ----------
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
    # returns (E, N, U) in meters
    if _GEO_KIND == 'pymap3d':
        pm = _GEO_LIB
        return pm.geodetic2enu(lat, lon, h, lat0, lon0, h0)
    elif _GEO_KIND == 'geographiclib':
        wgs84 = _GEO_LIB
        inv = wgs84.Inverse(lat0, lon0, lat, lon)
        s = inv['s12']
        azi = math.radians(inv['azi1'])
        n = s * math.cos(azi)
        e = s * math.sin(azi)
        u = h - h0
        return e, n, u
    else:
        R = 6378137.0
        dlat = math.radians(lat - lat0)
        dlon = math.radians(lon - lon0)
        latm = math.radians((lat + lat0) * 0.5)
        e = R * dlon * math.cos(latm)
        n = R * dlat
        u = h - h0
        return e, n, u

def euler_to_quat(roll, pitch, yaw) -> Quaternion:
    qx, qy, qz, qw = tf_transformations.quaternion_from_euler(roll, pitch, yaw)
    q = Quaternion()
    q.x, q.y, q.z, q.w = qx, qy, qz, qw
    return q

def quat_to_euler(q: Quaternion):
    return tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])  # roll,pitch,yaw

def wrap_pi(a):  # map to (-pi, pi]
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def E_rpy(roll, pitch):
    """Mapping from body rates [p,q,r] to Euler angle rates [roll_dot, pitch_dot, yaw_dot]."""
    cr = math.cos(roll);  sr = math.sin(roll)
    cp = math.cos(pitch); sp = math.sin(pitch)
    tp = math.tan(pitch)
    # [φ̇ θ̇ ψ̇]^T = T(φ,θ)[p q r]^T
    T = np.array([
        [1.0, sr*tp,  cr*tp],
        [0.0, cr,     -sr   ],
        [0.0, sr/cp,  cr/cp ]
    ])
    return T

@dataclass
class EKFParams:
    publish_tf: bool = True
    use_imu_yaw: bool = False           # yaw fusion is optional (compass/mag can be noisy)
    imu_accel_is_linear: bool = True    # True if IMU gives gravity-removed "linear acceleration"
    g: float = 9.80665

    # Process noise (per second); tune for your platform
    q_pos: float = 0.05                 # m^2/s
    q_vel: float = 1.0                  # (m/s)^2/s
    q_yaw: float = 0.05                 # rad^2/s
    q_pitch: float = 0.05               # rad^2/s
    q_roll: float = 0.05                # rad^2/s

    # Measurement noise
    r_gps_xy: float = 2.5               # m^2
    r_gps_z: float = 3.0                # m^2 (often noisier than horizontal)
    r_pitch_meas: float = 0.02          # rad^2
    r_roll_meas: float = 0.02           # rad^2
    r_yaw_meas: float = 0.3             # rad^2 (weak if enabled)

    # Topics/frames
    topic_imu: str = '/imu/data'
    topic_gps: str = '/fix'
    topic_odom: str = '/ekf/odom'
    frame_map: str = 'map'
    frame_base: str = 'base_link'

class EKFLocalisation3D(Node):
    def __init__(self):
        super().__init__('ekf_localisation_3d')

        # Parameters
        self.params = EKFParams(
            publish_tf = self.declare_parameter('publish_tf', True).get_parameter_value().bool_value,
            use_imu_yaw = self.declare_parameter('use_imu_yaw', False).get_parameter_value().bool_value,
            imu_accel_is_linear = self.declare_parameter('imu_accel_is_linear', True).get_parameter_value().bool_value,
            g = float(self.declare_parameter('g', 9.80665).value),
            q_pos = float(self.declare_parameter('q_pos', 0.05).value),
            q_vel = float(self.declare_parameter('q_vel', 1.0).value),
            q_yaw = float(self.declare_parameter('q_yaw', 0.05).value),
            q_pitch = float(self.declare_parameter('q_pitch', 0.05).value),
            q_roll = float(self.declare_parameter('q_roll', 0.05).value),
            r_gps_xy = float(self.declare_parameter('r_gps_xy', 2.5).value),
            r_gps_z = float(self.declare_parameter('r_gps_z', 3.0).value),
            r_pitch_meas = float(self.declare_parameter('r_pitch_meas', 0.02).value),
            r_roll_meas = float(self.declare_parameter('r_roll_meas', 0.02).value),
            r_yaw_meas = float(self.declare_parameter('r_yaw_meas', 0.3).value),
            topic_imu = self.declare_parameter('topic_imu', '/imu/data').get_parameter_value().string_value,
            topic_gps = self.declare_parameter('topic_gps', '/fix').get_parameter_value().string_value,
            topic_odom = self.declare_parameter('topic_odom', '/ekf/odom').get_parameter_value().string_value,
            frame_map = self.declare_parameter('frame_map', 'map').get_parameter_value().string_value,
            frame_base = self.declare_parameter('frame_base', 'base_link').get_parameter_value().string_value,
        )

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=20
        )
        self.sub_imu = self.create_subscription(Imu, self.params.topic_imu, self.imu_cb, qos)
        self.sub_gps = self.create_subscription(NavSatFix, self.params.topic_gps, self.gps_cb, qos)
        self.pub_odom = self.create_publisher(Odometry, self.params.topic_odom, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # State: [x y z vx vy vz yaw pitch roll]
        self.x = np.zeros((9,1))
        self.P = np.diag([25.0,25.0,25.0, 4.0,4.0,4.0, (math.pi**2), (math.pi**2)/4, (math.pi**2)/4])

        self.last_imu_time = None
        self.have_origin = False
        self.lat0 = self.lon0 = self.h0 = 0.0

        self.last_imu = None  # keep last IMU for twist/angular
        self.get_logger().info("EKF 3D localisation node started.")

    # ----------------- Callbacks -----------------
    def imu_cb(self, msg: Imu):
        t = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        if self.last_imu_time is None:
            self.last_imu_time = t
            # Initialize roll/pitch/yaw from IMU orientation if available
            if msg.orientation_covariance[0] >= 0.0:
                roll, pitch, yaw = quat_to_euler(msg.orientation)
                self.x[6,0] = wrap_pi(yaw)
                self.x[7,0] = wrap_pi(pitch)
                self.x[8,0] = wrap_pi(roll)
            self.last_imu = msg
            return

        dt = max(1e-3, t - self.last_imu_time)
        self.last_imu_time = t
        self.last_imu = msg

        # Body rates (p,q,r) and body accel (ax,ay,az)
        p = msg.angular_velocity.x
        q = msg.angular_velocity.y
        r = msg.angular_velocity.z
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z

        self.ekf_predict(dt, p, q, r, ax, ay, az)

        # Orientation measurement updates: roll & pitch (strong), yaw optional (weak)
        if msg.orientation_covariance[0] >= 0.0:
            roll_m, pitch_m, yaw_m = quat_to_euler(msg.orientation)
            self.ekf_scalar_angle_update(8, roll_m, self.params.r_roll_meas)   # roll
            self.ekf_scalar_angle_update(7, pitch_m, self.params.r_pitch_meas) # pitch
            if self.params.use_imu_yaw:
                self.ekf_scalar_angle_update(6, yaw_m, self.params.r_yaw_meas) # yaw (optional)

        # Publish at IMU rate
        self.publish_odom(msg.header.stamp)

    def gps_cb(self, msg: NavSatFix):
        if not self.have_origin:
            self.lat0, self.lon0, self.h0 = msg.latitude, msg.longitude, msg.altitude
            self.have_origin = True
            self.get_logger().info(f"Set ENU origin at lat={self.lat0:.8f}, lon={self.lon0:.8f}, h={self.h0:.2f} m")
        # Reject invalid fixes
        if msg.status.status is not None and msg.status.status < 0:
            return

        e, n, u = llh_to_enu(msg.latitude, msg.longitude, msg.altitude,
                             self.lat0, self.lon0, self.h0)
        z = np.array([[e],[n],[u]])

        # Build R from message covariance if present; else use params
        if len(msg.position_covariance) == 9 and msg.position_covariance[0] >= 0:
            Rx = msg.position_covariance[0]
            Rxy = msg.position_covariance[1]
            Ryx = msg.position_covariance[3]
            Ry = msg.position_covariance[4]
            Rz = msg.position_covariance[8] if msg.position_covariance[8] > 0 else self.params.r_gps_z
            R = np.array([[Rx, Rxy, 0.0],
                          [Ryx, Ry,  0.0],
                          [0.0, 0.0, Rz]])
        else:
            R = np.diag([self.params.r_gps_xy, self.params.r_gps_xy, self.params.r_gps_z])

        self.ekf_gps_update(z, R)
        self.publish_odom(self.get_clock().now().to_msg())

    # ----------------- EKF core -----------------
    def ekf_predict(self, dt, p, q, r, ax, ay, az):
        """
        x = [x y z vx vy vz yaw pitch roll]
        u = IMU body rates p,q,r (rad/s) and accel (m/s^2)
        """
        x = self.x.flatten()
        yaw, pitch, roll = x[6], x[7], x[8]

        # Angle rates from body rates
        T = E_rpy(roll, pitch)   # maps [p q r] -> [roll_dot, pitch_dot, yaw_dot]
        ang_rates = T @ np.array([[p],[q],[r]])
        roll_dot, pitch_dot, yaw_dot = ang_rates.flatten()

        # Rotation body->world (ENU) using Rz(yaw)*Ry(pitch)*Rx(roll)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)

        Rz = np.array([[cy, -sy, 0],
                       [sy,  cy, 0],
                       [0,    0, 1]])
        Ry = np.array([[ cp, 0, sp],
                       [  0, 1,  0],
                       [-sp, 0, cp]])
        Rx = np.array([[1,  0,   0],
                       [0, cr, -sr],
                       [0, sr,  cr]])
        R_b2w = Rz @ Ry @ Rx

        a_body = np.array([[ax],[ay],[az]])
        a_world = R_b2w @ a_body

        # Handle gravity depending on what IMU publishes
        if self.params.imu_accel_is_linear:
            # already gravity-removed
            g_world = np.array([[0],[0],[0]])
        else:
            # accel includes gravity; add gravity in world frame: a_world + g_world = true linear accel
            g_world = np.array([[0],[0],[-self.params.g]])
        a_world = a_world + g_world

        # Integrate
        vx = x[3] + a_world[0,0]*dt
        vy = x[4] + a_world[1,0]*dt
        vz = x[5] + a_world[2,0]*dt

        Xp = np.zeros((9,1))
        Xp[0,0] = x[0] + x[3]*dt + 0.5*a_world[0,0]*dt*dt
        Xp[1,0] = x[1] + x[4]*dt + 0.5*a_world[1,0]*dt*dt
        Xp[2,0] = x[2] + x[5]*dt + 0.5*a_world[2,0]*dt*dt
        Xp[3,0] = vx
        Xp[4,0] = vy
        Xp[5,0] = vz
        Xp[6,0] = wrap_pi(yaw + yaw_dot*dt)
        Xp[7,0] = wrap_pi(pitch + pitch_dot*dt)
        Xp[8,0] = wrap_pi(roll + roll_dot*dt)
        self.x = Xp

        # Jacobian F: we approximate with simple structure (constant-accel, angle coupling modest)
        F = np.eye(9)
        F[0,3] = dt; F[1,4] = dt; F[2,5] = dt
        # Very lightweight coupling from accel to position/velocity already handled via process noise.

        # Process noise
        qpos = self.params.q_pos * dt
        qvel = self.params.q_vel * dt
        qyaw = self.params.q_yaw * dt
        qpit = self.params.q_pitch * dt
        qroll= self.params.q_roll * dt
        Q = np.diag([qpos, qpos, qpos, qvel, qvel, qvel, qyaw, qpit, qroll])

        self.P = F @ self.P @ F.T + Q

    def ekf_gps_update(self, z, R):
        # h(x) = [x, y, z]
        H = np.zeros((3,9))
        H[0,0] = 1.0
        H[1,1] = 1.0
        H[2,2] = 1.0

        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(9) - K @ H) @ self.P

    def ekf_scalar_angle_update(self, idx, meas, r_var):
        """Update angle state x[idx] with wrapped innovation."""
        H = np.zeros((1,9))
        H[0, idx] = 1.0
        innov = wrap_pi(meas - self.x[idx,0])
        S = H @ self.P
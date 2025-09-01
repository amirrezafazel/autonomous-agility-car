#!/usr/bin/env python3
# ROS 2 Humble - EKF (3D) for IMU + GNSS
# State: [x, y, z, vx, vy, vz, yaw]
# Publishes local ENU odometry (x, y, z)

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
    q_pos: float = 0.05      # process noise on position
    q_vel: float = 0.5       # process noise on velocity
    q_yaw: float = 0.05      # process noise on yaw
    r_gps_pos: float = 2.5   # GPS measurement variance
    r_yaw_meas: float = 0.3  # IMU orientation yaw variance
    use_imu_orientation: bool = True  # enable yaw fusion
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
        self.P = np.diag([10, 10, 10, 5, 5, 5, (math.pi/2)])

        self.last_imu_time = None
        self.have_origin = False
        self.lat0 = self.lon0 = self.h0 = 0.0

        self.last_gz = 0.0
        self.get_logger().info("3D EKF localisation node started.")

    # ----------------- Callbacks -----------------
    def imu_cb(self, msg: Imu):
        t = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        if self.last_imu_time is None:
            self.last_imu_time = t
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

        # Prediction step
        self.ekf_predict(dt, ax, ay, az, gz)

        # Optional yaw measurement update
        if self.params.use_imu_orientation and msg.orientation_covariance[0] >= 0.0:
            meas_yaw = yaw_from_quaternion(msg.orientation)
            self.ekf_yaw_update(meas_yaw, self.params.r_yaw_meas)

        self.publish_odom(msg.header.stamp)

    def gps_cb(self, msg: GGA):
        if not self.have_origin:
            self.lat0, self.lon0, self.h0 = msg.latitude, msg.longitude, msg.altitude
            self.have_origin = True
            self.get_logger().info(f"Set ENU origin at lat={self.lat0:.8f}, lon={self.lon0:.8f}, h={self.h0:.2f} m")

        e, n, u = llh_to_enu(msg.latitude, msg.longitude, msg.altitude,
                             self.lat0, self.lon0, self.h0)
        z = np.array([[e], [n], [u]])
        R = np.eye(3) * self.params.r_gps_pos
        self.ekf_gps_update(z, R)
        self.publish_odom(self.get_clock().now().to_msg())

    # ----------------- EKF -----------------
    def ekf_predict(self, dt, ax, ay, az, gz):
        x, y, z, vx, vy, vz, yaw = self.x.flatten()

        cy = math.cos(yaw)
        sy = math.sin(yaw)
        ax_world = cy*ax - sy*ay
        ay_world = sy*ax + cy*ay
        az_world = az  # assume gravity-compensated

        # Motion model
        x_pred  = x + vx*dt + 0.5*ax_world*dt*dt
        y_pred  = y + vy*dt + 0.5*ay_world*dt*dt
        z_pred  = z + vz*dt + 0.5*az_world*dt*dt
        vx_pred = vx + ax_world*dt
        vy_pred = vy + ay_world*dt
        vz_pred = vz + az_world*dt
        yaw_pred = self.wrap_angle(yaw + gz*dt)

        self.x = np.array([[x_pred],[y_pred],[z_pred],[vx_pred],[vy_pred],[vz_pred],[yaw_pred]])

        # Process noise
        q = self.params
        Q = np.diag([q.q_pos*dt]*3 + [q.q_vel*dt]*3 + [q.q_yaw*dt])
        F = np.eye(7)
        F[0,3] = dt; F[1,4] = dt; F[2,5] = dt
        self.P = F @ self.P @ F.T + Q

    def ekf_gps_update(self, z, R):
        H = np.zeros((3,7))
        H[0,0] = 1; H[1,1] = 1; H[2,2] = 1
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ H) @ self.P
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
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = z
        odom.pose.pose.orientation = quaternion_from_yaw(yaw)

        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.linear.z = vz
        odom.twist.twist.angular.z = self.last_gz

        self.pub_odom.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self.params.frame_map
        tf.child_frame_id = self.params.frame_base
        tf.transform.translation.x = x
        tf.transform.translation.y = y
        tf.transform.translation.z = z
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
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, NavSatStatus
import serial
import pynmea2

# Convert GPS coordinates from the NMEA format to decimal degrees.
def nmea_latlon_to_decimal(value, dir: str, is_lat: bool):
    if value is None or value == "":
        return None
    v = float(value)
    if (is_lat and abs(v) > 90.0) or (not is_lat and abs(v) > 180.0):
        v_abs = abs(v)
        deg = int(v_abs // 100)
        minutes = v_abs - deg * 100
        dec = deg + minutes / 60.0
    else:
        dec = v
    if dir and dir.upper() in ('S', 'W'):
        dec = -abs(dec)
    return float(dec)

# Read GNSS GGA data from the serial port and publish them using NavSatFix msg type (which includes the covariance matrix).
class GNSSPublisher(Node):
    def __init__(self):
        super().__init__('gnss_publisher')
        self.declare_parameter('gnss_port', '/dev/ttyACM0')
        self.declare_parameter('gnss_baudrate', '19200')
        self.declare_parameter('frame_id', 'gps')
        # UERE estimate (m) - tune to receiver (normal ~3-10, RTK ~0.01-0.05)
        self.declare_parameter('uere', 5.0)
        # Vertical std factor relative to horizontal
        self.declare_parameter('vertical_factor', 2.0)
        # fallback horizontal std (m) if no HDOP available
        self.declare_parameter('fallback_horizontal_std', 10.0)
        self.publisher = self.create_publisher(NavSatFix, 'gnss/data', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        port = self.get_parameter('gnss_port').get_parameter_value().string_value
        baudrate = self.get_parameter('gnss_baudrate').get_parameter_value().string_value
        frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        uere = float(self.get_parameter('uere').get_parameter_value().double_value)
        vert_factor = float(self.get_parameter('vertical_factor').get_parameter_value().double_value)
        fallback_std = float(self.get_parameter('fallback_horizontal_std').get_parameter_value().double_value)

        try:
            with serial.Serial(port, baudrate, timeout=1) as ser:
                self.get_logger().info(f"Connected to {port} at {baudrate} baud.")
                while True:
                    line = ser.readline().decode('ascii', errors='ignore').strip()
                    # self.get_logger().info(f"{line}")
                    if not line:
                        continue
                    if not line.startswith('$'):
                        continue
                    try:
                        data = pynmea2.parse(line)
                    except pynmea2.ParseError:
                        continue

                    if isinstance(data, pynmea2.GGA):
                        lat = nmea_latlon_to_decimal(data.latitude, data.lat_dir, is_lat=True)
                        lon = nmea_latlon_to_decimal(data.longitude, data.lon_dir, is_lat=False)
                        alt = float(data.altitude) if data.altitude not in (None, "") else 0.0

                        msg = NavSatFix()
                        msg.header.stamp = self.get_clock().now().to_msg()
                        msg.header.frame_id = frame_id
                        msg.latitude = lat if lat is not None else 0.0
                        msg.longitude = lon if lon is not None else 0.0
                        msg.altitude = alt

                        # Status from gps_qual
                        try:
                            qual = int(data.gps_qual)
                        except Exception:
                            qual = 0
                        if qual == 0:
                            msg.status.status = NavSatStatus.STATUS_NO_FIX
                        else:
                            msg.status.status = NavSatStatus.STATUS_FIX

                        # Covariance calculation
                        hdop = float(data.horizontal_dil) if data.horizontal_dil not in (None, "") else None
                        num_sats = int(data.num_sats) if data.num_sats not in (None, "") else None

                        if hdop is not None and hdop > 0.0:
                            sigma_h = hdop * uere
                        else: # Fallback
                            sigma_h = fallback_std

                        # Scale UERE down a little if many satellites (heuristic).
                        if num_sats is not None and num_sats >= 8:
                            sigma_h = sigma_h * 0.9

                        var_h = max(1e-6, sigma_h * sigma_h)  # m^2, avoid exact zero
                        sigma_v = sigma_h * max(1.0, vert_factor)
                        var_v = sigma_v * sigma_v

                        # NavSatFix expects position_covariance in row-major 3x3 (ENU), using diagonal approximation: [var_east, 0, 0, 0, var_north, 0, 0, 0, var_up]
                        # Assume isotropic horizontal variance (east and north equal)
                        cov = [0.0 for _ in range(9)]
                        cov[0] = var_h  # east
                        cov[4] = var_h  # north
                        cov[8] = var_v  # up
                        msg.position_covariance = cov
                        msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN

                        self.publisher.publish(msg)
        except serial.SerialException as e:
            self.get_logger().error(f"Serial error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = GNSSPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
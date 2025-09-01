#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from custom_interfaces.msg import GGA
from std_msgs.msg import Header
import serial
import pynmea2

# Serial port and baudrate
PORT = '/dev/ttyACM0'
BAUDRATE = 19200

# Read GNSS GGA data from the serial port and publish them.
class GNSSPublisher(Node):
    def __init__(self):
        super().__init__('gnss_publisher')
        self.declare_parameter('gnss_port', '/dev/ttyACM0')
        self.publisher = self.create_publisher(GGA, 'gnss_data', 20)
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        PORT = self.get_parameter('gnss_port').get_parameter_value().string_value
        try:
            with serial.Serial(PORT, BAUDRATE, timeout=1) as ser:
                self.get_logger().info(f"Connected to {PORT} at {BAUDRATE} baud.")
                
                while True:
                    # Read and decode a line from the serial port
                    line = ser.readline().decode('ascii', errors='ignore').strip()
                    # Check if it is a valid NMEA sentence
                    if line.startswith('$'):
                        try:
                            data = pynmea2.parse(line)
                            # Only handle GGA sentences (position fix)
                            if isinstance(data, pynmea2.GGA):
                                msg = GGA()
                                msg.header.stamp = self.get_clock().now().to_msg()
                                msg.header.frame_id = 'base_link'
                                msg.utc_time = float(data.timestamp) if data.timestamp else 0.0
                                msg.latitude = float(data.latitude) if data.latitude else 0.0
                                msg.lat_dir = data.lat_dir if data.lat_dir else ""
                                msg.longitude = float(data.longitude) if data.longitude else 0.0
                                msg.lon_dir = data.lon_dir if data.lon_dir else ""
                                msg.gps_quality = int(data.gps_qual) if data.gps_qual else 0
                                msg.num_satellites = int(data.num_sats) if data.num_sats else 0
                                msg.hdop = float(data.horizontal_dil) if data.horizontal_dil else 0.0
                                msg.altitude = float(data.altitude) if data.altitude else 0.0
                                msg.altitude_units = data.altitude_units if data.altitude_units else ""
                                msg.geoid_separation = float(data.geo_sep) if data.geo_sep else 0.0
                                msg.geoid_separation_units = data.geo_sep_units if data.geo_sep_units else ""
                                msg.dgps_age = data.age_gps_data if data.age_gps_data else ""
                                msg.dgps_station_id = data.ref_station_id if data.ref_station_id else ""
                                self.publisher.publish(msg)
                        except pynmea2.ParseError:
                            # Skip malformed NMEA sentences
                            continue
        except serial.SerialException as e:
            print(f"Serial error: {e}")


def main(args=None):
    rclpy.init(args=args)
    gnss_publisher = GNSSPublisher()
    rclpy.spin(gnss_publisher)
    gnss_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
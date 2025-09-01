# GPS GGA Custom Message

# A header for timestamp and frame information
std_msgs/Header header

# Time of position fix, in seconds from NMEA timestamp
float64 utc_time

# Latitude, in degrees.
float64 latitude

# Direction of latitude: 'N' or 'S'
string lat_dir

# Longitude, in degrees.
float64 longitude

# Direction of longitude: 'E' or 'W'
string lon_dir

# GPS Quality Indicator
int32 gps_quality

# Number of satellites used in the fix
int32 num_satellites

# Horizontal Dilution of Precision
float64 hdop

# Antenna altitude relative to mean sea level (MSL)
float64 altitude

# Units of altitude, typically 'M' for meters
string altitude_units

# Geoid separation
float64 geoid_separation

# Units of geoid separation, typically 'M' for meters
string geoid_separation_units

# Age of differential GPS data (null if not using DGPS)
string dgps_age

# Differential reference station ID (null if not using DGPS)
string dgps_station_id
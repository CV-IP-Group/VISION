from rplidar import RPLidar

PORT = 'COM6'           # 连接端口号
BAUD = 115200           # 波特率，连接失败/2尝试
TIMEOUT = 1

lidar = RPLidar(PORT, baudrate=BAUD, timeout=TIMEOUT)
try:
    info = lidar.get_info()
    health = lidar.get_health()
    print('Info:', info)
    print('Health:', health)
finally:
    lidar.disconnect()

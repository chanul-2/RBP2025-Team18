#Final competition code
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
import numpy as np

def detect_monitor(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # 가장 큰 사각형을 모니터로 가정
    monitor_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(monitor_contour)

    # 화면 내부를 테두리에서 조금 떨어진 곳으로 잡음 제거
    margin = int(min(w, h) * 0.05)  # 화면 경계 5% 제거
    x, y, w, h = x + margin, y + margin, w - 2 * margin, h - 2 * margin

    if w * h >= 2000:
        monitor_region = image[y:y+h, x:x+w]
        return monitor_region
    else:
        return None

def get_dominant_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask_red1 = (hsv[:, :, 0] < 20) & (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 50)
    mask_red2 = (hsv[:, :, 0] > 160) & (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 50)
    mask_red = mask_red1 | mask_red2

    mask_green = (hsv[:, :, 0] > 40) & (hsv[:, :, 0] < 80) & (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 50)
    mask_blue = (hsv[:, :, 0] > 100) & (hsv[:, :, 0] < 140) & (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 50)

    count_red = np.sum(mask_red)
    count_green = np.sum(mask_green)
    count_blue = np.sum(mask_blue)

    total_pixels = count_red + count_green + count_blue
    if total_pixels == 0:
        return "N/A"

    red_ratio = count_red / total_pixels
    green_ratio = count_green / total_pixels
    blue_ratio = count_blue / total_pixels

    if red_ratio >= green_ratio and red_ratio >= blue_ratio:
        return "R"
    elif green_ratio >= red_ratio and green_ratio >= blue_ratio:
        return "G"
    else:
        return "B"

class DetermineColor(Node):
    def __init__(self):
        super().__init__('color_detector')
        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.callback, 10)
        self.color_pub = self.create_publisher(Header, '/rotate_cmd', 10)
        self.bridge = CvBridge()
        self.count = 0

    def callback(self, data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')

            msg = Header()
            msg = data.header
            msg.frame_id = '0'  # default: STOP

            c = get_dominant_color(image)
            print(c)
            if c == 'R':
                msg.frame_id = '-1'   # CCW
            elif c == 'B':
                msg.frame_id = '1'  # CW
            elif c == 'G':
                msg.frame_id = '0'   # STOP

            self.color_pub.publish(msg)

        except CvBridgeError as e:
            self.get_logger().error('Failed to convert image: %s' % e)

if __name__ == '__main__':
    rclpy.init()
    detector = DetermineColor()
    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()





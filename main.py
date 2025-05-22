#Final competition code
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
import numpy as np

import cv2
import numpy as np

def improved_dominant_color_in_monitor(image):
    # 1) 이미지 크기
    h, w = image.shape[:2]

    # 2) HSV 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    # 3) R/G/B 마스크 생성
    bright = (v_ch > 50)
    r_m = (((h_ch < 20) | (h_ch > 160)) & (s_ch > 50) & (v_ch > 50)).astype(np.uint8) * 255
    g_m = ((h_ch > 40) & (h_ch < 80) & (s_ch > 50) & (v_ch > 50)).astype(np.uint8) * 255
    b_m = ((h_ch > 100) & (h_ch < 140) & (s_ch > 50) & (v_ch > 50)).astype(np.uint8) * 255
    mask_rgb = (bright & ((r_m > 0) | (g_m > 0) | (b_m > 0))).astype(np.uint8) * 255

    # 4) 화면 영역 컨투어 검출 + 유효 영역(2000px 이상)
    cnts, _ = cv2.findContours(mask_rgb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in cnts if cv2.contourArea(c) > 2000]
    if not valid:
        return None

    # 5) 유효한 컨투어 병합 → 최소 회전 사각형 추출
    all_pts = np.vstack(valid)
    hull = cv2.convexHull(all_pts)
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect).astype(np.int32)

    # 6) 모니터 마스크 생성
    monitor_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(monitor_mask, [box], -1, 255, thickness=-1)

    # 7) R/G/B 픽셀 수 계산 (모니터 내부만)
    R = int(cv2.countNonZero(((r_m > 0) & (monitor_mask > 0)).astype(np.uint8)))
    G = int(cv2.countNonZero(((g_m > 0) & (monitor_mask > 0)).astype(np.uint8)))
    B = int(cv2.countNonZero(((b_m > 0) & (monitor_mask > 0)).astype(np.uint8)))

    # 8) 주된 색 결정
    dominant = max([("R", R), ("G", G), ("B", B)], key=lambda x: x[1])[0]

    return dominant

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

            c = improved_dominant_color_in_monitor(image)
            if c == 'R':
                msg.frame_id = '-1'   # CCW
            elif c == 'B':
                msg.frame_id = '1'  # CW

            self.color_pub.publish(msg)

        except CvBridgeError as e:
            self.get_logger().error('Failed to convert image: %s' % e)

if __name__ == '__main__':
    rclpy.init()
    detector = DetermineColor()
    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()





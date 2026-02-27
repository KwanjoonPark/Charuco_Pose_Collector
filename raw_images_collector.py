#!/usr/bin/env python
"""
RealSense 카메라로 RGB + Depth 이미지 캡처

사용법:
    python raw_images_collector.py [--output OUTPUT_DIR] [--frames NUM_FRAMES]

키 조작:
    's' 또는 스페이스: 한 장 캡처 (사진 모드)
    'b': 연속 녹화 시작
    'e': 연속 녹화 종료
    'q': 종료
"""

import rospy
import cv2
import numpy as np
import os
import argparse
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError


class FrameCapture:
    def __init__(self, output_dir="captured_frames", max_frames=50):
        rospy.init_node('frame_capture', anonymous=True)

        self.output_dir = output_dir
        self.max_frames = max_frames
        self.frame_count = 0

        # 디렉토리 생성
        self.rgb_dir = os.path.join(output_dir, "rgb")
        self.depth_dir = os.path.join(output_dir, "depth")
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)

        self.bridge = CvBridge()

        # 카메라 매트릭스
        self.camera_matrix = None
        self.camera_matrix_saved = False

        # 현재 프레임
        self.current_rgb = None
        self.current_depth = None

        # 녹화 상태
        self.is_recording = False

        # ROS 토픽 구독
        self.rgb_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        self.info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.info_callback)

        print(f"출력 디렉토리: {output_dir}")
        print(f"목표 프레임 수: {max_frames}")
        print("\n키 조작:")
        print("  's' 또는 스페이스: 한 장 캡처 (사진 모드)")
        print("  'b': 연속 녹화 시작")
        print("  'e': 연속 녹화 종료")
        print("  'q': 종료")
        print("\n카메라 대기 중...")

    def info_callback(self, msg):
        if not self.camera_matrix_saved:
            self.camera_matrix = np.array(msg.K).reshape(3, 3)

            # cam_K.txt 저장
            cam_k_path = os.path.join(self.output_dir, "cam_K.txt")
            np.savetxt(cam_k_path, self.camera_matrix, fmt='%.6f')
            print(f"카메라 매트릭스 저장: {cam_k_path}")
            self.camera_matrix_saved = True

    def rgb_callback(self, msg):
        try:
            self.current_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"RGB 변환 오류: {e}")

    def depth_callback(self, msg):
        try:
            self.current_depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        except CvBridgeError as e:
            rospy.logerr(f"Depth 변환 오류: {e}")

    def save_frame(self):
        if self.current_rgb is None or self.current_depth is None:
            return False

        filename = f"{self.frame_count:06d}.png"
        cv2.imwrite(os.path.join(self.rgb_dir, filename), self.current_rgb)
        cv2.imwrite(os.path.join(self.depth_dir, filename), self.current_depth)

        self.frame_count += 1
        print(f"[{self.frame_count}/{self.max_frames}] 저장 완료")

        if self.frame_count >= self.max_frames:
            print(f"\n{self.max_frames}프레임 캡처 완료!")
            self.is_recording = False
            return True

        return False

    def run(self):
        rate = rospy.Rate(50)

        while not rospy.is_shutdown():
            if self.current_rgb is not None:
                # 화면 표시
                display = self.current_rgb.copy()

                # 상태 표시
                if self.is_recording:
                    status_text = f"RECORDING [{self.frame_count}/{self.max_frames}]"
                    status_color = (0, 0, 255)  # 빨간색
                else:
                    status_text = f"Ready [{self.frame_count}/{self.max_frames}]"
                    status_color = (0, 255, 0)  # 초록색

                cv2.putText(display, status_text, (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                cv2.putText(display, "'s'/Space:Capture  'b':Record  'e':Stop  'q':Quit", (10, display.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

                cv2.imshow("Frame Capture", display)

                # 녹화 중이면 프레임 저장
                if self.is_recording:
                    self.save_frame()

            key = cv2.waitKey(1) & 0xFF

            # 's' 또는 스페이스: 한 장 캡처
            if key == ord('s') or key == 32:  # 32 = 스페이스바
                if not self.is_recording:
                    self.save_frame()
            elif key == ord('b'):
                if not self.is_recording:
                    print("연속 녹화 시작")
                    self.is_recording = True
            elif key == ord('e'):
                if self.is_recording:
                    print("연속 녹화 종료")
                    self.is_recording = False
            elif key == ord('q'):
                print("종료")
                break

            rate.sleep()

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='RealSense RGB+Depth 프레임 캡처')
    parser.add_argument('--output', '-o', type=str, default='captured_frames', help='출력 디렉토리')
    parser.add_argument('--frames', '-f', type=int, default=50, help='캡처할 프레임 수')
    args = parser.parse_args()

    try:
        capture = FrameCapture(output_dir=args.output, max_frames=args.frames)
        capture.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()

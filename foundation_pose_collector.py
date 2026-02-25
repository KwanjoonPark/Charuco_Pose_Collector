#!/usr/bin/env python3
"""
FoundationPose 데이터 수집기
ChArUco 보드를 사용하여 카메라 포즈를 추정하고 FoundationPose run_demo.py에 필요한 데이터 구조로 저장

데이터 구조:
test_scene_dir/
├── rgb/
│   ├── 000000.png
│   └── ...
├── depth/
│   ├── 000000.png      # 16-bit PNG (mm 단위)
│   └── ...
├── masks/
│   └── 000000.png      # 첫 프레임 마스크만 필수
├── cam_K.txt           # 3x3 카메라 intrinsic
├── ob_in_cam/
│   ├── 000000.txt      # 4x4 변환 행렬
│   └── ...
└── model/
    └── model.obj       # 3D 메쉬 (사용자가 직접 준비)

사용법:
    python foundation_pose_collector.py [--output OUTPUT_DIR] [--start INDEX]

키 조작:
    'b': 녹화 시작
    'e': 녹화 종료
    's': 현재 프레임 수동 저장
    'z': 마지막 저장 취소 (undo)
    'r': 카운터 리셋
    'q': 종료
"""

import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
import os
import argparse
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from charuco_config import (
    CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y,
    CHARUCO_SQUARE_LENGTH, CHARUCO_MARKER_LENGTH,
    ARUCO_DICT_TYPE,
    CHARUCO_OFFSET_X, CHARUCO_OFFSET_Y, CHARUCO_OFFSET_Z
)


class FoundationPoseCollector:
    def __init__(self, output_dir="test_scene", start_index=0):
        rospy.init_node('foundation_pose_collector', anonymous=True)

        # 출력 디렉토리 설정
        self.output_dir = output_dir
        self.save_counter = start_index

        # 디렉토리 생성
        self.rgb_dir = os.path.join(output_dir, "rgb")
        self.depth_dir = os.path.join(output_dir, "depth")
        self.masks_dir = os.path.join(output_dir, "masks")
        self.ob_in_cam_dir = os.path.join(output_dir, "ob_in_cam")
        self.model_dir = os.path.join(output_dir, "model")

        for d in [self.rgb_dir, self.depth_dir, self.masks_dir, self.ob_in_cam_dir, self.model_dir]:
            os.makedirs(d, exist_ok=True)

        # ChArUco 보드 설정 (charuco_config.py에서 import)
        self.SQUARES_X = CHARUCO_SQUARES_X
        self.SQUARES_Y = CHARUCO_SQUARES_Y
        self.SQUARE_LENGTH = CHARUCO_SQUARE_LENGTH
        self.MARKER_LENGTH = CHARUCO_MARKER_LENGTH
        self.ARUCO_DICT = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)

        # 보드가 정방향으로 달려있음
        self.BOARD_FLIPPED = True

        # Offset 설정 (charuco_config.py에서 import, 직관적 좌표계 기준, 단위: 미터)
        # 화면 표시 좌표계 (사용자 입력용):
        #   X: 오른쪽 방향 (+)
        #   Y: 위쪽 방향 (+)  ← 직관적
        #   Z: 카메라에서 멀어지는 방향 (+)
        self.OFFSET_X = CHARUCO_OFFSET_X
        self.OFFSET_Y = CHARUCO_OFFSET_Y
        self.OFFSET_Z = CHARUCO_OFFSET_Z

        # ArUco Detector 설정 (OpenCV 버전 호환)
        self.use_new_aruco_api = False
        try:
            self.parameters = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.ARUCO_DICT, self.parameters)
            self.use_new_aruco_api = True
            print("New Aruco API 사용 가능")
        except AttributeError:
            self.parameters = cv2.aruco.DetectorParameters_create()
            print("Old Aruco API 사용")

        self.parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self.parameters.minMarkerPerimeterRate = 0.01
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 23
        self.parameters.adaptiveThreshWinSizeStep = 10
        self.parameters.cornerRefinementWinSize = 5

        # ChArUco 보드 생성 (OpenCV 버전 호환)
        self.use_charuco_detector = False
        try:
            self.board = cv2.aruco.CharucoBoard(
                (self.SQUARES_X, self.SQUARES_Y),
                self.SQUARE_LENGTH,
                self.MARKER_LENGTH,
                self.ARUCO_DICT
            )
            self.board.setLegacyPattern(True)
            self.charuco_detector = cv2.aruco.CharucoDetector(self.board)
            self.use_charuco_detector = True
            print("CharucoDetector 사용 가능 (OpenCV 4.7+)")
        except AttributeError:
            self.board = cv2.aruco.CharucoBoard_create(
                self.SQUARES_X, self.SQUARES_Y,
                self.SQUARE_LENGTH, self.MARKER_LENGTH,
                self.ARUCO_DICT
            )
            self.charuco_detector = None
            print("CharucoBoard_create 사용 (구버전)")

        # ROS 설정
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None

        # 현재 프레임 데이터
        self.current_image = None
        self.current_depth = None
        self.current_rvec = None
        self.current_tvec = None
        self.current_tvec_offset = None  # offset 적용된 tvec
        self.pose_valid = False
        self.is_recording = False # 녹화 상태 플래그
        self.max_frames = 200 # 자동 종료를 위한 최대 프레임 수

        # Undo 기능을 위한 히스토리
        self.save_history = []

        # ROS 토픽 구독
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        self.info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.info_callback)

        print(f"\n출력 디렉토리: {output_dir}")
        print(f"시작 인덱스: {start_index}")
        print("\n키 조작:")
        print("  'b': 녹화 시작")
        print("  'e': 녹화 종료")
        print("  's': 현재 프레임 수동 저장")
        print("  'z': 마지막 저장 취소 (undo)")
        print("  'r': 카운터 리셋")
        print("  'q': 종료")
        print(f"  녹화는 {self.max_frames} 프레임에 도달하면 자동으로 종료됩니다.")
        print("\nWaiting for camera...")

    def info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.dist_coeffs = np.array(msg.D)

            # cam_K.txt 저장
            cam_k_path = os.path.join(self.output_dir, "cam_K.txt")
            np.savetxt(cam_k_path, self.camera_matrix, fmt='%.6f')
            print(f"Camera intrinsic 저장: {cam_k_path}")
            print(f"Camera Matrix:\n{self.camera_matrix}")

    def depth_callback(self, msg):
        try:
            self.current_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
        except CvBridgeError as e:
            rospy.logerr(f"Depth conversion error: {e}")

    def image_callback(self, msg):
        if self.camera_matrix is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        self.current_image = cv_image.copy()
        display_image = cv_image.copy()
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # ChArUco 감지
        if self.use_charuco_detector:
            charuco_corners, charuco_ids, corners, ids = self.charuco_detector.detectBoard(gray)
        else:
            if self.use_new_aruco_api:
                corners, ids, rejected = self.detector.detectMarkers(gray)
            else:
                corners, ids, rejected = aruco.detectMarkers(gray, self.ARUCO_DICT, parameters=self.parameters)

            if ids is not None and len(corners) > 0:
                retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                    corners, ids, gray, self.board
                )
            else:
                charuco_corners, charuco_ids = None, None

        # 마커 표시
        if ids is not None:
            aruco.drawDetectedMarkers(display_image, corners, ids)

        self.pose_valid = False
        self.current_rvec = None
        self.current_tvec = None
        self.current_tvec_offset = None

        if charuco_corners is not None and len(charuco_corners) > 0:
            # 포즈 추정
            if self.use_charuco_detector:
                obj_points, img_points = self.board.matchImagePoints(charuco_corners, charuco_ids)
                if obj_points is not None and len(obj_points) >= 6:
                    valid, rvec, tvec = cv2.solvePnP(
                        obj_points, img_points, self.camera_matrix, self.dist_coeffs
                    )
                else:
                    valid = False
            else:
                valid, rvec, tvec = aruco.estimatePoseCharucoBoard(
                    charuco_corners, charuco_ids, self.board,
                    self.camera_matrix, self.dist_coeffs, None, None
                )

            if valid:
                self.pose_valid = True
                self.current_tvec = tvec

                # 회전 행렬
                R_raw, _ = cv2.Rodrigues(rvec)

                # 보드가 거꾸로 달린 경우 보정 (X, Y축 180도 회전)
                if self.BOARD_FLIPPED:
                    # 180도 회전 행렬: X, Y 반전, Z 유지
                    R_flip = np.array([
                        [-1,  0,  0],
                        [ 0, -1,  0],
                        [ 0,  0,  1]
                    ], dtype=np.float64)
                    R = R_raw @ R_flip
                    # 보정된 rvec 계산
                    rvec_corrected, _ = cv2.Rodrigues(R)
                    self.current_rvec = rvec_corrected
                else:
                    R = R_raw
                    self.current_rvec = rvec

                # Offset 적용 (직관적 좌표계 -> 카메라 좌표계)
                # 사용자 입력은 Y+ = 위쪽, 카메라 좌표계는 Y+ = 아래쪽이므로 Y 반전
                offset_in_board = np.array([self.OFFSET_X, -self.OFFSET_Y, self.OFFSET_Z])
                offset_in_camera = R @ offset_in_board
                tvec_offset = tvec.flatten() + offset_in_camera
                self.current_tvec_offset = tvec_offset

                # Yaw, Pitch, Roll 계산 (보정된 R 사용)
                sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
                singular = sy < 1e-6

                if not singular:
                    pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
                    yaw = np.degrees(np.arctan2(-R[2, 0], sy))
                    roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
                else:
                    pitch = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
                    yaw = np.degrees(np.arctan2(-R[2, 0], sy))
                    roll = 0

                # Pitch 정규화 (보드가 카메라를 향할 때 0도)
                if pitch > 90:
                    pitch = pitch - 180
                elif pitch < -90:
                    pitch = pitch + 180

                # 보드 원점에 축 그리기 (보정된 좌표계)
                cv2.drawFrameAxes(display_image, self.camera_matrix, self.dist_coeffs,
                                  self.current_rvec, tvec, 0.03)

                # Offset 위치에 파란색 점 그리기
                offset_point_3d = np.array([[tvec_offset[0], tvec_offset[1], tvec_offset[2]]], dtype=np.float32)
                offset_pixel_2d, _ = cv2.projectPoints(offset_point_3d, np.zeros(3), np.zeros(3),
                                                       self.camera_matrix, self.dist_coeffs)
                px = int(float(offset_pixel_2d[0][0][0]))
                py = int(float(offset_pixel_2d[0][0][1]))
                cv2.circle(display_image, (px, py), 10, (255, 0, 0), -1)  # 파란색 점
                cv2.circle(display_image, (px, py), 12, (255, 255, 255), 2)  # 흰색 테두리

                # 포즈 정보 표시 (Y축 반전: 위쪽이 +)
                pos_cm = tvec.flatten() * 100
                offset_pos_cm = tvec_offset * 100

                cv2.putText(display_image, f"Board(cm): X={pos_cm[0]:.1f}, Y={-pos_cm[1]:.1f}, Z={pos_cm[2]:.1f}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_image, f"Angle: Pitch={pitch:.1f}, Yaw={yaw:.1f}, Roll={roll:.1f}",
                           (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display_image, f"Target(cm): X={offset_pos_cm[0]:.1f}, Y={-offset_pos_cm[1]:.1f}, Z={offset_pos_cm[2]:.1f}",
                           (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(display_image, f"Offset: ({self.OFFSET_X*100:.1f}, {self.OFFSET_Y*100:.1f}, {self.OFFSET_Z*100:.1f})cm",
                           (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # 녹화 및 상태 표시 로직
        if self.is_recording:
            if self.pose_valid:
                self.save_frame()
                status_text = "RECORDING"
                status_color = (0, 165, 255)  # 주황색
            else:
                status_text = "RECORDING (No Marker)"
                status_color = (0, 0, 255)  # 빨간색
        else:
            status_color = (0, 255, 0) if self.pose_valid else (0, 0, 255)
            status_text = "Ready" if self.pose_valid else "No Marker"

        cv2.putText(display_image, f"[{self.save_counter:06d}] {status_text}", (10, display_image.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        cv2.imshow("FoundationPose Collector", display_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('b'):
            if not self.is_recording:
                print("녹화 시작")
                self.is_recording = True
        elif key == ord('e'):
            if self.is_recording:
                print("녹화 종료")
                self.is_recording = False
        elif key == ord('s'):
            self.save_frame()
        elif key == ord('z'):
            self.undo_last_save()
        elif key == ord('r'):
            self.reset_counter()
        elif key == ord('q'):
            rospy.signal_shutdown("User quit")

    def save_frame(self):
        """현재 프레임 저장"""
        if self.current_image is None:
            if not self.is_recording: print("이미지 없음")
            return

        if not self.pose_valid:
            if not self.is_recording: print("포즈 감지 안됨 - 마커를 확인하세요")
            return

        if self.current_depth is None:
            if not self.is_recording: print("깊이 이미지 없음")
            return

        filename = f"{self.save_counter:06d}"

        # RGB 저장
        rgb_path = os.path.join(self.rgb_dir, f"{filename}.png")
        cv2.imwrite(rgb_path, self.current_image)

        # Depth 저장 (16-bit PNG, mm 단위 - RealSense 기본값)
        depth_path = os.path.join(self.depth_dir, f"{filename}.png")
        cv2.imwrite(depth_path, self.current_depth)

        # ob_in_cam 저장 (4x4 변환 행렬)
        # ob_in_cam = 카메라 좌표계에서 본 물체의 포즈
        # offset 적용된 위치 사용
        R, _ = cv2.Rodrigues(self.current_rvec)
        T_ob_in_cam = np.eye(4)
        T_ob_in_cam[:3, :3] = R
        T_ob_in_cam[:3, 3] = self.current_tvec_offset  # offset 적용된 위치

        ob_in_cam_path = os.path.join(self.ob_in_cam_dir, f"{filename}.txt")
        np.savetxt(ob_in_cam_path, T_ob_in_cam, fmt='%.6f')

        # 히스토리에 추가 (undo용)
        self.save_history.append({
            'index': self.save_counter,
            'rgb': rgb_path,
            'depth': depth_path,
            'ob_in_cam': ob_in_cam_path
        })

        if not self.is_recording:
            print(f"[{filename}] 수동 저장 완료 - RGB, Depth, ob_in_cam")
        else: # Recording is active, print message for automatic save
            print(f"[{filename}] 자동 저장 완료 - RGB, Depth, ob_in_cam")
        
        self.save_counter += 1

        # 프레임 수 제한 초과 시 녹화 자동 종료
        if self.is_recording and self.save_counter >= self.max_frames:
            self.is_recording = False
            print(f"최대 프레임 수({self.max_frames})에 도달하여 녹화를 자동 종료합니다.")

    def undo_last_save(self):
        """마지막 저장 취소"""
        if self.is_recording:
            print("녹화 중에는 취소할 수 없습니다.")
            return

        if not self.save_history:
            print("취소할 저장이 없습니다")
            return

        last = self.save_history.pop()

        # 파일 삭제
        for key in ['rgb', 'depth', 'ob_in_cam']:
            if os.path.exists(last[key]):
                os.remove(last[key])

        self.save_counter = last['index']
        print(f"[{last['index']:06d}] 저장 취소됨")

    def reset_counter(self):
        """카운터 리셋"""
        if self.is_recording:
            print("녹화 중에는 리셋할 수 없습니다.")
            return
        self.save_counter = 0
        self.save_history.clear()
        print("카운터 리셋됨")

    def run(self):
        rospy.spin()


def main():
    parser = argparse.ArgumentParser(description='FoundationPose 데이터 수집기')
    parser.add_argument('--output', '-o', type=str, default='test_scene',
                       help='출력 디렉토리 (기본값: test_scene)')
    parser.add_argument('--start', '-s', type=int, default=0,
                       help='시작 인덱스 (기본값: 0)')
    args = parser.parse_args()

    try:
        collector = FoundationPoseCollector(
            output_dir=args.output,
            start_index=args.start
        )
        collector.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

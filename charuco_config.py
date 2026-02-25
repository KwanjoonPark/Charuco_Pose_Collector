"""
ChArUco 보드 전역 설정
모든 스크립트에서 이 파일을 import하여 사용
"""
import cv2

# ChArUco 보드 크기
CHARUCO_SQUARES_X = 4
CHARUCO_SQUARES_Y = 5
CHARUCO_SQUARE_LENGTH = 0.030  # 3.0 cm (미터 단위)
CHARUCO_MARKER_LENGTH = 0.022  # 2.2 cm (미터 단위)

# ArUco 딕셔너리 타입 (OpenCV 상수)
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_250

# 오프셋: 보드 원점 → 목표 지점 (미터 단위, 기본값 0)
CHARUCO_OFFSET_X = 0.0
CHARUCO_OFFSET_Y = 0.0
CHARUCO_OFFSET_Z = 0.0

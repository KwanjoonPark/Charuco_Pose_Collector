# 6-DoF Pose Estimation

ChArUco 마커를 이용한 6자유도(6-DoF) 포즈 추정 시스템입니다. Intel RealSense 카메라와 ROS를 사용하여 실시간 또는 배치 처리로 3D 위치(x, y, z)와 회전(pitch, yaw, roll)을 계산합니다.

## 주요 기능

- ChArUco 마커 기반 6-DoF 포즈 추정
- ROS 실시간 처리 및 ROS bag 배치 처리
- FoundationPose 학습용 데이터 수집
- Edge 기반 세그멘테이션 (FloodFill + 다각형 근사)
- ChArUco 마커 제거 (단색 채우기, 인페인팅)

## 요구 사항

### 하드웨어
- Intel RealSense D435/D415 카메라

### 소프트웨어
```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate

# Python 패키지 설치
pip install -r requirements.txt

# ROS 패키지 (ROS Noetic/Melodic)
sudo apt install ros-noetic-cv-bridge ros-noetic-sensor-msgs python3-rosbag
```

## 파일 구조

```
├── charuco_config.py              # ChArUco 보드 전역 설정 (모든 스크립트 공통)
├── charuco_pose_estimator.py      # 실시간 포즈 추정 (ROS 노드)
├── rosbag_pose_estimator.py       # ROS bag 배치 처리
├── foundation_pose_collector.py   # FoundationPose 데이터 수집기
├── raw_images_collector.py        # RGB + Depth 이미지 캡처
├── edge_based_segmentation.py     # Edge 기반 세그멘테이션
├── segmentation_debug_steps.py    # 세그멘테이션 단계별 시각화
├── requirements.txt               # Python 의존성
└── opencv_charuco_remover/
    ├── grab_solid.py              # 단색 채우기로 마커 제거
    └── grab_inpainting.py         # 인페인팅으로 마커 제거
```

## 사용 방법

### 1. 실시간 포즈 추정

```bash
python charuco_pose_estimator.py
```
- 's' 키: 현재 프레임과 포즈 데이터 저장
- Ctrl+C: 종료

### 2. ROS Bag 배치 처리

```bash
python rosbag_pose_estimator.py --bag /path/to/file.bag --out pose_data.csv
```
bag 파일에서 모든 프레임을 추출하고 포즈를 계산합니다.

### 3. FoundationPose 데이터 수집

```bash
python foundation_pose_collector.py [--output OUTPUT_DIR] [--start INDEX]
```
ChArUco 보드를 사용하여 카메라 포즈를 추정하고 FoundationPose `run_demo.py`에 필요한 데이터 구조(rgb, depth, ob_in_cam, cam_K.txt)로 저장합니다.

- 'b' 키: 연속 녹화 시작
- 'e' 키: 연속 녹화 종료
- 's' 키: 현재 프레임 수동 저장
- 'z' 키: 마지막 저장 취소 (undo)
- 'r' 키: 카운터 리셋
- 'q' 키: 종료

### 4. RGB + Depth 이미지 캡처

```bash
python raw_images_collector.py [--output OUTPUT_DIR] [--frames NUM_FRAMES]
```
마커 감지 없이 RealSense 카메라에서 RGB + Depth 이미지를 캡처합니다.

- 's' 또는 스페이스: 한 장 캡처
- 'b' 키: 연속 녹화 시작
- 'e' 키: 연속 녹화 종료
- 'q' 키: 종료

### 5. 마커 제거

```bash
# 단색으로 마커 영역 채우기
python opencv_charuco_remover/grab_solid.py

# 인페인팅으로 마커 영역 채우기
python opencv_charuco_remover/grab_inpainting.py
```

### 6. 세그멘테이션

```bash
# Edge 기반 세그멘테이션 (Canny + FloodFill + 다각형 근사)
python edge_based_segmentation.py

# 단계별 시각화 디버깅
python segmentation_debug_steps.py
```

## ChArUco 보드 설정

모든 보드 설정은 **`charuco_config.py`** 한 곳에서 관리됩니다. 보드를 교체하면 이 파일만 수정하면 됩니다.

```python
# charuco_config.py
CHARUCO_SQUARES_X = 4           # 보드 가로(X) 방향 사각형 개수
CHARUCO_SQUARES_Y = 5           # 보드 세로(Y) 방향 사각형 개수
CHARUCO_SQUARE_LENGTH = 0.030   # 체스보드 사각형 한 변의 길이 (미터)
CHARUCO_MARKER_LENGTH = 0.022   # ArUco 마커 한 변의 길이 (미터)
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_250  # ArUco 딕셔너리 타입

# 오프셋: 보드 원점 → 목표 지점 (미터)
CHARUCO_OFFSET_X = 0.0          # 오른쪽 방향 (+)
CHARUCO_OFFSET_Y = 0.0          # 위쪽 방향 (+)
CHARUCO_OFFSET_Z = 0.0          # 카메라에서 멀어지는 방향 (+)
```

> **주의**: `SQUARE_LENGTH`과 `MARKER_LENGTH`는 **실제 인쇄된 보드를 자로 측정한 값**을 입력해야 합니다. 프린터에 따라 축소/확대될 수 있으므로 반드시 실측하세요.

### ArUco 딕셔너리

기본값은 `DICT_4X4_250` (4x4 비트 패턴, 250개 고유 마커)입니다. 딕셔너리를 변경하면 보드를 새로 인쇄해야 합니다.

## 출력 데이터

### CSV 형식
```csv
Timestamp, Image_File, Distance(cm), Pitch(deg), Yaw(deg), Roll(deg), X(cm), Y(cm), Z(cm)
```

## 알고리즘 개요

### 포즈 추정
1. ChArUco 코너 검출 (서브픽셀 정밀도)
2. `estimatePoseCharucoBoard()`로 PnP 풀이
3. 회전 행렬 → 오일러 각도 변환 (ZYX 순서)
4. 오프셋 변환 적용 (마커 → 목표 지점)

### 마커 제거 (grab_solid.py)
1. GrabCut으로 마커 영역 마스크 생성
2. 마스크 외부에서 단일 색상 샘플링
3. 단색으로 마스크 영역 채우기
4. Gaussian Blur로 경계 부드럽게 블렌딩

### 마커 제거 (grab_inpainting.py)
1. GrabCut으로 마커 영역 마스크 생성
2. OpenCV `inpaint()` (TELEA 알고리즘)으로 주변 색상 참조하여 채우기

### Edge 기반 세그멘테이션
1. Canny 엣지 검출 → Morphology Closing → Dilation으로 방파제 생성
2. 중심점 + 코너에서 FloodFill 시작점 결정 (LAB 색공간 비교)
3. FloodFill로 객체 영역 채우기 (방파제가 경계 역할)
4. Convex Hull → 다각형 근사 (4/5/6각형 우선순위)

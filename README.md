# 6-DoF Pose Estimation

ChArUco 마커를 이용한 6자유도(6-DoF) 포즈 추정 시스템입니다. Intel RealSense 카메라와 ROS를 사용하여 실시간 또는 배치 처리로 3D 위치(x, y, z)와 회전(pitch, yaw, roll)을 계산합니다.

## 주요 기능

- ChArUco 마커 기반 6-DoF 포즈 추정
- ROS 실시간 처리 및 ROS bag 배치 처리
- Edge 기반 세그멘테이션 (FloodFill + 다각형 근사)
- ChArUco 마커 제거 (단색 채우기, 인페인팅)

## 요구 사항

### 하드웨어
- Intel RealSense D435/D415 카메라

### 소프트웨어
```bash
# Python 패키지
pip install opencv-python>=4.7 numpy

# ROS 패키지 (ROS Noetic/Melodic)
sudo apt install ros-noetic-cv-bridge ros-noetic-sensor-msgs
```

## 사용 방법

### 1. 실시간 포즈 추정

```bash
python charuco_pose_estimator.py
```
- 's' 키: 현재 프레임과 포즈 데이터 저장
- 'q' 키: 종료

### 2. ROS Bag 배치 처리

```bash
python rosbag_pose_estimator.py
```
bag 파일에서 모든 프레임을 추출하고 포즈를 계산합니다.

### 3. 마커 제거

```bash
# 단색으로 마커 영역 채우기
python opencv_charuco_remover/grab_solid.py

# 인페인팅으로 마커 영역 채우기
python opencv_charuco_remover/grab_inpainting.py
```

### 4. 세그멘테이션

```bash
# Edge 기반 세그멘테이션 (Canny + FloodFill + 다각형 근사)
python edge_based_segmentation.py
```

### 5. 디버깅

```bash
# edge_based_segmentation 단계별 시각화
python segmentation_debug_steps.py
```

## 파일 구조

```
├── charuco_pose_estimator.py       # 실시간 포즈 추정 (ROS 노드)
├── rosbag_pose_estimator.py        # ROS bag 배치 처리
├── edge_based_segmentation.py      # Edge 기반 세그멘테이션
├── segmentation_debug_steps.py     # 세그멘테이션 디버깅
└── opencv_charuco_remover/
    ├── grab_solid.py               # 단색 채우기로 마커 제거
    └── grab_inpainting.py          # 인페인팅으로 마커 제거
```

## ChArUco 보드 설정

### 설정 파라미터 설명

| 파라미터 | 설명 | 단위 |
|---------|------|------|
| `SQUARES_X` | 보드 가로(X) 방향 사각형 개수 | 개 |
| `SQUARES_Y` | 보드 세로(Y) 방향 사각형 개수 | 개 |
| `SQUARE_LENGTH` | 체스보드 사각형 한 변의 길이 | 미터(m) |
| `MARKER_LENGTH` | ArUco 마커 한 변의 길이 (사각형보다 작아야 함) | 미터(m) |
| `ARUCO_DICT` | ArUco 딕셔너리 타입 | - |

> **주의**: `SQUARE_LENGTH`과 `MARKER_LENGTH`는 **실제 인쇄된 보드를 자로 측정한 값**을 입력해야 합니다. 프린터에 따라 축소/확대될 수 있으므로 반드시 실측하세요.

### 오프셋(Offset) 파라미터

마커 보드의 원점(좌측 하단)에서 실제 추정하고자 하는 목표 지점까지의 거리를 설정합니다.

| 파라미터 | 설명 | 단위 |
|---------|------|------|
| `OFFSET_X` | 오른쪽 방향 (+) | 미터(m) |
| `OFFSET_Y` | 위쪽 방향 (+) | 미터(m) |
| `OFFSET_Z` | 카메라에서 멀어지는 방향 (+) | 미터(m) |

### 설정 파일별 안내

프로젝트에는 **두 가지 설정 방식**이 존재합니다:

#### 방식 1: 공통 설정 파일 — `charuco_config.py`

```python
# charuco_config.py
CHARUCO_SQUARES_X = 4
CHARUCO_SQUARES_Y = 5
CHARUCO_SQUARE_LENGTH = 0.030  # 3.0 cm (미터 단위)
CHARUCO_MARKER_LENGTH = 0.022  # 2.2 cm (미터 단위)
ARUCO_DICT_TYPE = "DICT_4X4_250"
```

이 파일을 import하여 사용하는 스크립트:
- `foundation_pose_collector.py` — FoundationPose 데이터 수집
- `opencv_charuco_remover/grab_solid.py` — 단색 채우기 마커 제거
- `opencv_charuco_remover/grab_solid_fast.py` — 단색 채우기 (고속)
- `opencv_charuco_remover/grab_inpainting.py` — 인페인팅 마커 제거

**이 스크립트들의 보드 크기를 변경하려면 `charuco_config.py`만 수정하면 됩니다.**

#### 방식 2: 스크립트 내부 직접 설정

아래 스크립트들은 각 파일 상단에 설정값이 직접 정의되어 있습니다. 보드를 교체하면 **각 파일을 개별적으로 수정**해야 합니다.

| 스크립트 | 설정 위치 | 현재 값 |
|---------|----------|--------|
| `charuco_pose_estimator.py` | 17~21행 (클래스 `__init__` 내부) | 5x4, 2.0cm / 1.5cm |
| `rosbag_pose_estimator.py` | 14~18행 (모듈 상단 상수) | 4x5, 2.0cm / 1.5cm |
| `opencv_charuco_remover.py` | 16~19행 (모듈 상단 상수) | 5x4, 2.0cm / 1.5cm |
| `color_diff_marker.py` | 22~25행 (모듈 상단 상수) | 5x4, 2.0cm / 1.5cm |
| `grab_cut_marker.py` | 20~23행 (모듈 상단 상수) | 5x4, 2.0cm / 1.5cm |

**설정 변경 예시** (`rosbag_pose_estimator.py` 14~24행):

```python
# ==========================================
# Configuration: ChArUco Board Parameters
# ==========================================
SQUARES_X = 4          # 가로 사각형 수
SQUARES_Y = 5          # 세로 사각형 수
SQUARE_LENGTH = 0.020  # 사각형 크기: 20mm → 0.020m
MARKER_LENGTH = 0.015  # 마커 크기: 15mm → 0.015m
ARUCO_DICT_TYPE = aruco.DICT_4X4_250

# 오프셋: 보드 원점 → 목표 지점
OFFSET_X = 0.16   # 오른쪽 방향 (미터)
OFFSET_Y = 0.07   # 위쪽 방향 (미터)
OFFSET_Z = -0.02  # 앞쪽 방향 (미터)
# ==========================================
```

### SQUARES_X와 SQUARES_Y 방향 주의

> **주의**: `SQUARES_X`와 `SQUARES_Y`의 정의가 스크립트마다 다를 수 있습니다.
> - `charuco_pose_estimator.py`: `SQUARES_X=5`, `SQUARES_Y=4` (가로 5, 세로 4)
> - `rosbag_pose_estimator.py`: `SQUARES_X=4`, `SQUARES_Y=5` (가로 4, 세로 5)
>
> 보드를 세로로 세워서 사용하는지, 가로로 놓는지에 따라 X/Y가 뒤바뀔 수 있습니다. **실제 보드 방향에 맞게 값을 확인**하세요.

### ArUco 딕셔너리

모든 스크립트에서 `DICT_4X4_250`을 사용합니다. 이는 4x4 비트 패턴에 250개의 고유 마커를 포함합니다. OpenCV에서 사용 가능한 다른 옵션:
- `DICT_4X4_50`, `DICT_4X4_100`, `DICT_4X4_250`, `DICT_4X4_1000`
- `DICT_5X5_*`, `DICT_6X6_*`, `DICT_7X7_*`

**딕셔너리를 변경하면 보드를 새로 인쇄해야 합니다.**

## 출력 데이터

### CSV 형식
```csv
Timestamp, Image_File, Distance(cm), Pitch(deg), Yaw(deg), Roll(deg), X(cm), Y(cm), Z(cm)
```

### 출력 디렉토리
```
├── images/                          # RGB 이미지
├── depth/                           # Depth 이미지
├── self_pose.csv                    # 포즈 데이터
└── dataset/
    ├── final_solid_color/           # 단색 채우기 결과
    ├── final_stable_inpainting/     # 인페인팅 결과
    └── inpainting_background_off/   # 배경 제거 결과
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

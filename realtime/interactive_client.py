#!/usr/bin/env python3
"""
Interactive FoundationPose Client
실시간으로 마스크만 표시하다가 'p' 키를 누르면 Pose Estimation 수행.

Usage:
    # 웹캠 사용
    python interactive_client.py --server_ip localhost --source webcam

    # 테스트 이미지 사용
    python interactive_client.py --server_ip localhost --source dataset --test_dir ../vcb/ref_views/test_scene

    # RealSense 사용 (pyrealsense2 필요)
    python interactive_client.py --server_ip localhost --source realsense

Controls:
    p: Pose Estimation 수행
    m: Mask Only 모드로 돌아가기
    s: 현재 프레임 저장
    q: 종료
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import time
from pathlib import Path
from typing import Optional, Generator, Tuple
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
import zmq


class Mode(Enum):
    MASK_ONLY = "mask_only"
    POSE = "pose"


@dataclass
class FrameData:
    """프레임 데이터."""
    color: np.ndarray
    depth: Optional[np.ndarray] = None
    K: Optional[np.ndarray] = None
    frame_id: int = 0


class FrameSource:
    """프레임 소스 베이스 클래스."""

    def __init__(self):
        self.K = None

    def get_frame(self) -> Optional[FrameData]:
        raise NotImplementedError

    def release(self):
        pass


class WebcamSource(FrameSource):
    """웹캠 소스."""

    def __init__(self, device_id: int = 0):
        super().__init__()
        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open webcam {device_id}")

        # 기본 카메라 파라미터 (추정값)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fx = fy = 615.0 * (w / 640)
        self.K = np.array([
            [fx, 0, w / 2],
            [0, fy, h / 2],
            [0, 0, 1]
        ], dtype=np.float32)

        self.frame_id = 0

    def get_frame(self) -> Optional[FrameData]:
        ret, frame = self.cap.read()
        if not ret:
            return None
        self.frame_id += 1
        return FrameData(color=frame, K=self.K, frame_id=self.frame_id)

    def release(self):
        self.cap.release()


class DatasetSource(FrameSource):
    """테스트 데이터셋 소스."""

    def __init__(self, test_dir: str, loop: bool = True):
        super().__init__()
        self.test_dir = Path(test_dir)
        self.loop = loop

        # 이미지 파일 목록
        self.rgb_files = sorted((self.test_dir / 'rgb').glob('*.png'))
        if not self.rgb_files:
            raise RuntimeError(f"No images found in {self.test_dir / 'rgb'}")

        self.depth_dir = self.test_dir / 'depth'
        self.current_idx = 0

        # Camera intrinsics
        cam_file = self.test_dir / 'cam_K.txt'
        if cam_file.exists():
            self.K = np.loadtxt(str(cam_file), dtype=np.float32)
        else:
            self.K = np.array([[615, 0, 320], [0, 615, 240], [0, 0, 1]], dtype=np.float32)

    def get_frame(self) -> Optional[FrameData]:
        if self.current_idx >= len(self.rgb_files):
            if self.loop:
                self.current_idx = 0
            else:
                return None

        rgb_file = self.rgb_files[self.current_idx]
        color = cv2.imread(str(rgb_file))

        depth = None
        depth_file = self.depth_dir / rgb_file.name
        if depth_file.exists():
            depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)

        frame_data = FrameData(
            color=color,
            depth=depth,
            K=self.K,
            frame_id=self.current_idx
        )
        self.current_idx += 1
        return frame_data


class RealSenseSource(FrameSource):
    """RealSense 카메라 소스."""

    def __init__(self):
        super().__init__()
        try:
            import pyrealsense2 as rs
        except ImportError:
            raise RuntimeError("pyrealsense2 not installed. Run: pip install pyrealsense2")

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        profile = self.pipeline.start(config)

        # Camera intrinsics
        color_profile = profile.get_stream(rs.stream.color)
        intr = color_profile.as_video_stream_profile().get_intrinsics()
        self.K = np.array([
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1]
        ], dtype=np.float32)

        self.align = rs.align(rs.stream.color)
        self.frame_id = 0

    def get_frame(self) -> Optional[FrameData]:
        import pyrealsense2 as rs

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame:
            return None

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data()) if depth_frame else None

        self.frame_id += 1
        return FrameData(color=color, depth=depth, K=self.K, frame_id=self.frame_id)

    def release(self):
        self.pipeline.stop()


class InteractiveClient:
    """인터랙티브 클라이언트."""

    def __init__(
        self,
        server_ip: str,
        port: int,
        source: FrameSource,
        jpeg_quality: int = 80
    ):
        self.server_ip = server_ip
        self.port = port
        self.source = source
        self.jpeg_quality = jpeg_quality

        self.mode = Mode.MASK_ONLY
        self.last_pose_result = None
        self.save_count = 0

        self._init_zmq()
        self.logger = logging.getLogger(__name__)

    def _init_zmq(self):
        """ZeroMQ 초기화."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 10000)  # 10s timeout
        self.socket.setsockopt(zmq.SNDTIMEO, 2000)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{self.server_ip}:{self.port}")

    def send_request(self, frame: FrameData, mode: Mode) -> dict:
        """서버에 요청 전송."""
        # JPEG 압축
        _, color_jpg = cv2.imencode(
            '.jpg', frame.color,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )

        data = {
            'color': color_jpg.tobytes(),
            'K': frame.K.flatten().tolist() if frame.K is not None else None,
        }

        if frame.depth is not None:
            data['depth'] = frame.depth.astype(np.uint16).tobytes()
            data['depth_shape'] = frame.depth.shape

        # 모드에 따라 명령 설정
        if mode == Mode.MASK_ONLY:
            data['command'] = 'mask_only'

        try:
            self.socket.send_pyobj(data)
            result = self.socket.recv_pyobj()
            return result
        except zmq.error.Again:
            return {'success': False, 'error': 'Server timeout'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def decode_visualization(self, result: dict) -> Optional[np.ndarray]:
        """서버에서 받은 시각화 이미지 디코딩."""
        if 'visualization' not in result:
            return None

        vis_bytes = result['visualization']
        vis_arr = np.frombuffer(vis_bytes, dtype=np.uint8)
        return cv2.imdecode(vis_arr, cv2.IMREAD_COLOR)

    def create_status_bar(self, vis: np.ndarray, result: dict) -> np.ndarray:
        """하단 상태 바 추가."""
        h, w = vis.shape[:2]
        bar_height = 40
        bar = np.zeros((bar_height, w, 3), dtype=np.uint8)

        # 모드 표시
        mode_text = f"Mode: {self.mode.value.upper()}"
        mode_color = (0, 255, 255) if self.mode == Mode.MASK_ONLY else (0, 255, 0)
        cv2.putText(bar, mode_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

        # Latency
        latency = result.get('latency_ms', 0)
        cv2.putText(bar, f"Latency: {latency:.0f}ms", (200, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 단축키
        cv2.putText(bar, "[P]ose [M]ask [S]ave [Q]uit", (w - 280, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return np.vstack([vis, bar])

    def save_frame(self, vis: np.ndarray, result: dict):
        """현재 프레임 저장."""
        save_dir = Path("./saved_frames")
        save_dir.mkdir(exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{self.mode.value}_{self.save_count:04d}.jpg"

        cv2.imwrite(str(save_dir / filename), vis)
        self.logger.info(f"Saved: {save_dir / filename}")
        self.save_count += 1

    def run(self):
        """메인 루프."""
        self.logger.info("Interactive Client Started")
        self.logger.info("Controls: [P] Pose, [M] Mask only, [S] Save, [Q] Quit")

        cv2.namedWindow("FoundationPose Interactive", cv2.WINDOW_NORMAL)

        try:
            while True:
                # 프레임 가져오기
                frame = self.source.get_frame()
                if frame is None:
                    self.logger.warning("No frame available")
                    time.sleep(0.1)
                    continue

                # 서버에 요청
                t_start = time.time()
                result = self.send_request(frame, self.mode)
                t_total = time.time() - t_start

                # 시각화
                vis = self.decode_visualization(result)
                if vis is None:
                    vis = frame.color.copy()
                    cv2.putText(vis, f"Error: {result.get('error', 'No visualization')}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Pose 결과 저장
                if self.mode == Mode.POSE and result.get('success'):
                    self.last_pose_result = result

                # 상태 바 추가
                vis_with_bar = self.create_status_bar(vis, result)

                # 표시
                cv2.imshow("FoundationPose Interactive", vis_with_bar)

                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('p'):
                    self.mode = Mode.POSE
                    self.logger.info("Switched to POSE mode")
                elif key == ord('m'):
                    self.mode = Mode.MASK_ONLY
                    self.logger.info("Switched to MASK_ONLY mode")
                elif key == ord('s'):
                    self.save_frame(vis, result)

                # Dataset 모드에서는 약간의 딜레이
                if isinstance(self.source, DatasetSource):
                    time.sleep(0.1)

        except KeyboardInterrupt:
            self.logger.info("Interrupted")
        finally:
            cv2.destroyAllWindows()
            self.source.release()
            self.socket.close()
            self.context.term()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(
        description='Interactive FoundationPose Client',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--server_ip', type=str, default='localhost',
                        help='Server IP address')
    parser.add_argument('--port', type=int, default=5555,
                        help='Server port')
    parser.add_argument('--source', type=str, default='dataset',
                        choices=['webcam', 'dataset', 'realsense'],
                        help='Frame source')
    parser.add_argument('--test_dir', type=str, default='vcb/ref_views/test_scene',
                        help='Test dataset directory (for dataset source)')
    parser.add_argument('--webcam_id', type=int, default=0,
                        help='Webcam device ID')
    parser.add_argument('--jpeg_quality', type=int, default=80,
                        help='JPEG compression quality')

    args = parser.parse_args()

    # 프레임 소스 생성
    if args.source == 'webcam':
        source = WebcamSource(args.webcam_id)
    elif args.source == 'dataset':
        source = DatasetSource(args.test_dir, loop=True)
    elif args.source == 'realsense':
        source = RealSenseSource()
    else:
        raise ValueError(f"Unknown source: {args.source}")

    # 클라이언트 실행
    client = InteractiveClient(
        server_ip=args.server_ip,
        port=args.port,
        source=source,
        jpeg_quality=args.jpeg_quality
    )
    client.run()


if __name__ == '__main__':
    main()

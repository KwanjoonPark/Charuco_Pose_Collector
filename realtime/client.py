#!/usr/bin/env python3
"""
FoundationPose ZeroMQ Client
Local PC에서 실행 - RealSense 카메라로 이미지를 캡처하여 서버로 전송

Usage:
    # RealSense 카메라 사용
    python client.py --server_ip 192.168.1.100 --port 5555

    # 이미지 파일로 테스트
    python client.py --server_ip 192.168.1.100 --test_image ./test.png

    # 비디오 파일 사용
    python client.py --server_ip 192.168.1.100 --video ./test.mp4
"""

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
import zmq


@dataclass
class ClientConfig:
    """클라이언트 설정."""
    server_ip: str = 'localhost'
    port: int = 5555
    width: int = 640
    height: int = 480
    fps: int = 30
    jpeg_quality: int = 80
    show_visualization: bool = True
    save_results: bool = False
    output_dir: str = './results'


class RealSenseCamera:
    """RealSense 카메라 래퍼."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        try:
            import pyrealsense2 as rs
        except ImportError:
            raise ImportError(
                "pyrealsense2 not installed. Install with: pip install pyrealsense2"
            )

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Color stream
        self.config.enable_stream(
            rs.stream.color, width, height, rs.format.bgr8, fps
        )
        # Depth stream
        self.config.enable_stream(
            rs.stream.depth, width, height, rs.format.z16, fps
        )

        # Align depth to color
        self.align = rs.align(rs.stream.color)

        # Start pipeline
        profile = self.pipeline.start(self.config)

        # Get camera intrinsics
        color_profile = profile.get_stream(rs.stream.color)
        intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        self.K = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ], dtype=np.float32)

        self.width = width
        self.height = height
        logging.info(f"RealSense initialized: {width}x{height}@{fps}fps")
        logging.info(f"Camera K:\n{self.K}")

    def get_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """RGB와 Depth 프레임 획득."""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        return color, depth

    def stop(self):
        """카메라 정지."""
        self.pipeline.stop()


class WebCamera:
    """일반 웹캠 래퍼."""

    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480):
        self.cap = cv2.VideoCapture(device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Default intrinsics (approximate)
        self.K = np.array([
            [615, 0, width / 2],
            [0, 615, height / 2],
            [0, 0, 1]
        ], dtype=np.float32)

        self.width = width
        self.height = height
        logging.info(f"WebCamera initialized: {width}x{height}")

    def get_frames(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """RGB 프레임 획득 (depth 없음)."""
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        return frame, None

    def stop(self):
        """카메라 정지."""
        self.cap.release()


class PoseEstimationClient:
    """ZeroMQ 기반 Pose Estimation 클라이언트."""

    def __init__(self, config: ClientConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self._init_zmq()

        self.frame_count = 0
        self.total_latency = 0

    def _init_zmq(self):
        """ZeroMQ 소켓 초기화."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 10000)  # 10s timeout
        self.socket.setsockopt(zmq.SNDTIMEO, 5000)   # 5s timeout

        server_addr = f"tcp://{self.config.server_ip}:{self.config.port}"
        self.socket.connect(server_addr)
        self.logger.info(f"Connected to server: {server_addr}")

    def ping(self) -> bool:
        """서버 연결 확인."""
        try:
            self.socket.send_pyobj({'command': 'ping'})
            response = self.socket.recv_pyobj()
            self.logger.info(f"Server status: {response}")
            return response.get('status') == 'ok'
        except zmq.error.Again:
            self.logger.error("Server not responding")
            return False

    def send_frame(
        self,
        color: np.ndarray,
        depth: Optional[np.ndarray] = None,
        K: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """프레임 전송 및 결과 수신."""
        # JPEG 압축
        _, color_jpg = cv2.imencode(
            '.jpg', color,
            [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
        )

        # 데이터 준비
        data = {
            'color': color_jpg.tobytes(),
            'timestamp': time.time(),
        }

        if K is not None:
            data['K'] = K.flatten().tolist()

        if depth is not None:
            data['depth'] = depth.astype(np.uint16).tobytes()
            data['depth_shape'] = depth.shape

        # 전송 및 수신
        t_send = time.time()
        self.socket.send_pyobj(data)
        result = self.socket.recv_pyobj()
        t_recv = time.time()

        # 네트워크 지연 추가
        result['network_ms'] = (t_recv - t_send) * 1000 - result.get('latency_ms', 0)

        self.frame_count += 1
        if result.get('success'):
            self.total_latency += result.get('latency_ms', 0)

        return result

    def visualize(
        self,
        color: np.ndarray,
        result: Dict[str, Any]
    ) -> np.ndarray:
        """결과 시각화."""
        vis = color.copy()

        if not result.get('success'):
            cv2.putText(vis, f"Error: {result.get('error', 'Unknown')}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return vis

        # 정보 텍스트
        trans = result.get('translation', [0, 0, 0])
        euler = result.get('euler_angles', {})
        latency = result.get('latency_ms', 0)
        network = result.get('network_ms', 0)
        fps_avg = result.get('fps_avg', 0)

        texts = [
            f"X: {trans[0]*100:+6.2f} cm",
            f"Y: {trans[1]*100:+6.2f} cm",
            f"Z: {trans[2]*100:+6.2f} cm",
            f"Roll:  {euler.get('roll', 0):+7.2f} deg",
            f"Pitch: {euler.get('pitch', 0):+7.2f} deg",
            f"Yaw:   {euler.get('yaw', 0):+7.2f} deg",
            f"",
            f"Latency: {latency:.1f}ms (net: {network:.1f}ms)",
            f"Server FPS: {fps_avg:.1f}",
        ]

        y = 25
        for text in texts:
            cv2.putText(vis, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 25

        return vis

    def run_camera(self, camera_type: str = 'realsense'):
        """카메라 스트리밍 모드."""
        # 카메라 초기화
        if camera_type == 'realsense':
            camera = RealSenseCamera(
                self.config.width, self.config.height, self.config.fps
            )
        else:
            camera = WebCamera(0, self.config.width, self.config.height)

        # 서버 연결 확인
        if not self.ping():
            self.logger.error("Cannot connect to server")
            return

        self.logger.info("Starting camera stream. Press 'q' to quit.")

        try:
            while True:
                # 프레임 획득
                color, depth = camera.get_frames()

                # 서버로 전송
                result = self.send_frame(color, depth, camera.K)

                # 시각화
                if self.config.show_visualization:
                    vis = self.visualize(color, result)
                    cv2.imshow('FoundationPose Client', vis)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s') and result.get('success'):
                        # 결과 저장
                        self._save_result(color, result)

                # 로그
                if self.frame_count % 30 == 0:
                    self.logger.info(
                        f"Frame {self.frame_count}: "
                        f"latency={result.get('latency_ms', 0):.1f}ms"
                    )

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        finally:
            camera.stop()
            cv2.destroyAllWindows()

    def run_image(self, image_path: str):
        """단일 이미지 테스트."""
        color = cv2.imread(image_path)
        if color is None:
            self.logger.error(f"Cannot read image: {image_path}")
            return

        # 서버 연결 확인
        if not self.ping():
            return

        result = self.send_frame(color)
        self.logger.info(f"Result: {result}")

        if self.config.show_visualization:
            vis = self.visualize(color, result)
            cv2.imshow('Result', vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def run_video(self, video_path: str):
        """비디오 파일 처리."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Cannot open video: {video_path}")
            return

        # 서버 연결 확인
        if not self.ping():
            return

        self.logger.info(f"Processing video: {video_path}")

        while True:
            ret, color = cap.read()
            if not ret:
                break

            result = self.send_frame(color)

            if self.config.show_visualization:
                vis = self.visualize(color, result)
                cv2.imshow('Video', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    def _save_result(self, color: np.ndarray, result: Dict[str, Any]):
        """결과 저장."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime('%Y%m%d_%H%M%S')

        # 이미지 저장
        cv2.imwrite(str(output_dir / f'{timestamp}_color.png'), color)

        # Pose 저장
        if result.get('pose'):
            pose = np.array(result['pose'])
            np.savetxt(str(output_dir / f'{timestamp}_pose.txt'), pose)

        self.logger.info(f"Saved result to {output_dir}/{timestamp}_*")

    def shutdown_server(self):
        """서버 종료 명령."""
        try:
            self.socket.send_pyobj({'command': 'shutdown'})
            response = self.socket.recv_pyobj()
            self.logger.info(f"Server response: {response}")
        except:
            pass

    def cleanup(self):
        """리소스 정리."""
        self.socket.close()
        self.context.term()


def parse_args():
    parser = argparse.ArgumentParser(description='FoundationPose ZeroMQ Client')
    parser.add_argument('--server_ip', type=str, default='localhost',
                        help='GPU server IP address')
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--jpeg_quality', type=int, default=80,
                        help='JPEG compression quality (1-100)')

    # 입력 소스
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--realsense', action='store_true', default=True,
                        help='Use RealSense camera (default)')
    source.add_argument('--webcam', action='store_true',
                        help='Use webcam')
    source.add_argument('--test_image', type=str,
                        help='Test with single image')
    source.add_argument('--video', type=str,
                        help='Process video file')

    # 기타
    parser.add_argument('--no_vis', action='store_true',
                        help='Disable visualization')
    parser.add_argument('--save', action='store_true',
                        help='Save results')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--shutdown', action='store_true',
                        help='Send shutdown command to server')

    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    args = parse_args()

    config = ClientConfig(
        server_ip=args.server_ip,
        port=args.port,
        width=args.width,
        height=args.height,
        fps=args.fps,
        jpeg_quality=args.jpeg_quality,
        show_visualization=not args.no_vis,
        save_results=args.save,
        output_dir=args.output_dir,
    )

    client = PoseEstimationClient(config)

    try:
        if args.shutdown:
            client.shutdown_server()
        elif args.test_image:
            client.run_image(args.test_image)
        elif args.video:
            client.run_video(args.video)
        elif args.webcam:
            client.run_camera('webcam')
        else:
            client.run_camera('realsense')
    finally:
        client.cleanup()


if __name__ == '__main__':
    main()

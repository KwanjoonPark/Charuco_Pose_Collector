#!/usr/bin/env python3
"""
FoundationPose ROS Node
ROS 토픽에서 이미지를 받아 ZeroMQ 서버로 전송하고 결과를 발행.

Usage:
    rosrun foundation_pose ros_client.py _server_ip:=192.168.1.100 _port:=5555

Topics:
    Subscribed:
        /camera/color/image_raw (sensor_msgs/Image)
        /camera/aligned_depth_to_color/image_raw (sensor_msgs/Image) [optional]
        /camera/color/camera_info (sensor_msgs/CameraInfo)

    Published:
        /foundation_pose/pose (geometry_msgs/PoseStamped)
        /foundation_pose/pose_array (geometry_msgs/PoseArray)
        /foundation_pose/transform (geometry_msgs/TransformStamped)
        /foundation_pose/visualization (sensor_msgs/Image)
        /foundation_pose/status (std_msgs/String)
"""

import sys
import threading
import time
from typing import Optional, Dict, Any

import cv2
import numpy as np
import zmq

try:
    import rospy
    from std_msgs.msg import String, Header
    from sensor_msgs.msg import Image, CameraInfo
    from geometry_msgs.msg import PoseStamped, PoseArray, Pose, TransformStamped
    from cv_bridge import CvBridge, CvBridgeError
    import tf2_ros
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("ROS not available. Install ROS and source setup.bash")


class FoundationPoseROSNode:
    """FoundationPose ROS 노드."""

    def __init__(self):
        rospy.init_node('foundation_pose_node', anonymous=True)

        # Parameters
        self.server_ip = rospy.get_param('~server_ip', 'localhost')
        self.port = rospy.get_param('~port', 5555)
        self.jpeg_quality = rospy.get_param('~jpeg_quality', 80)
        self.publish_tf = rospy.get_param('~publish_tf', True)
        self.frame_id = rospy.get_param('~frame_id', 'camera_color_optical_frame')
        self.object_frame_id = rospy.get_param('~object_frame_id', 'object')
        self.rate_limit = rospy.get_param('~rate_limit', 10.0)  # Hz

        # State
        self.bridge = CvBridge()
        self.K = None
        self.latest_depth = None
        self.last_process_time = 0
        self.lock = threading.Lock()

        # ZeroMQ
        self._init_zmq()

        # TF Broadcaster
        if self.publish_tf:
            self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Publishers
        self.pose_pub = rospy.Publisher(
            '~pose', PoseStamped, queue_size=1
        )
        self.pose_array_pub = rospy.Publisher(
            '~pose_array', PoseArray, queue_size=1
        )
        self.vis_pub = rospy.Publisher(
            '~visualization', Image, queue_size=1
        )
        self.status_pub = rospy.Publisher(
            '~status', String, queue_size=1
        )

        # Subscribers
        rospy.Subscriber(
            '/camera/color/image_raw', Image,
            self.color_callback, queue_size=1, buff_size=2**24
        )
        rospy.Subscriber(
            '/camera/aligned_depth_to_color/image_raw', Image,
            self.depth_callback, queue_size=1, buff_size=2**24
        )
        rospy.Subscriber(
            '/camera/color/camera_info', CameraInfo,
            self.camera_info_callback, queue_size=1
        )

        rospy.loginfo(f"FoundationPose ROS node initialized")
        rospy.loginfo(f"  Server: {self.server_ip}:{self.port}")
        rospy.loginfo(f"  Rate limit: {self.rate_limit} Hz")
        rospy.loginfo(f"  Frame ID: {self.frame_id}")

    def _init_zmq(self):
        """ZeroMQ 소켓 초기화."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5s timeout
        self.socket.setsockopt(zmq.SNDTIMEO, 2000)  # 2s timeout
        self.socket.setsockopt(zmq.LINGER, 0)

        server_addr = f"tcp://{self.server_ip}:{self.port}"
        self.socket.connect(server_addr)
        rospy.loginfo(f"Connected to ZeroMQ server: {server_addr}")

        # Ping test
        try:
            self.socket.send_pyobj({'command': 'ping'})
            response = self.socket.recv_pyobj()
            rospy.loginfo(f"Server status: {response.get('status', 'unknown')}")
        except zmq.error.Again:
            rospy.logwarn("Server not responding to ping")

    def camera_info_callback(self, msg: CameraInfo):
        """카메라 정보 콜백."""
        if self.K is None:
            self.K = np.array(msg.K, dtype=np.float32).reshape(3, 3)
            rospy.loginfo(f"Camera intrinsics received:\n{self.K}")

    def depth_callback(self, msg: Image):
        """Depth 이미지 콜백."""
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        except CvBridgeError as e:
            rospy.logwarn(f"Depth conversion error: {e}")

    def color_callback(self, msg: Image):
        """Color 이미지 콜백 - 메인 처리."""
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_process_time < 1.0 / self.rate_limit:
            return

        if self.K is None:
            rospy.logwarn_throttle(5, "Waiting for camera_info...")
            return

        with self.lock:
            self.last_process_time = current_time

            try:
                # Convert image
                color = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            except CvBridgeError as e:
                rospy.logerr(f"Color conversion error: {e}")
                return

            # Process
            result = self._send_to_server(color, self.latest_depth)

            if result is None:
                self._publish_status("Server communication error")
                return

            if not result.get('success'):
                self._publish_status(f"Detection failed: {result.get('error', 'unknown')}")
                return

            # Publish results
            self._publish_pose(result, msg.header)
            self._publish_visualization(color, result, msg.header)
            self._publish_status(
                f"OK - Latency: {result.get('latency_ms', 0):.0f}ms, "
                f"FPS: {result.get('fps_avg', 0):.1f}"
            )

    def _send_to_server(
        self,
        color: np.ndarray,
        depth: Optional[np.ndarray]
    ) -> Optional[Dict[str, Any]]:
        """서버로 이미지 전송."""
        try:
            # JPEG 압축
            _, color_jpg = cv2.imencode(
                '.jpg', color,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            )

            data = {
                'color': color_jpg.tobytes(),
                'K': self.K.flatten().tolist(),
            }

            if depth is not None:
                data['depth'] = depth.astype(np.uint16).tobytes()
                data['depth_shape'] = depth.shape

            self.socket.send_pyobj(data)
            result = self.socket.recv_pyobj()
            return result

        except zmq.error.Again:
            rospy.logwarn("Server timeout")
            self._reconnect()
            return None
        except Exception as e:
            rospy.logerr(f"Communication error: {e}")
            self._reconnect()
            return None

    def _reconnect(self):
        """서버 재연결."""
        rospy.loginfo("Reconnecting to server...")
        try:
            self.socket.close()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.setsockopt(zmq.RCVTIMEO, 5000)
            self.socket.setsockopt(zmq.SNDTIMEO, 2000)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.connect(f"tcp://{self.server_ip}:{self.port}")
        except Exception as e:
            rospy.logerr(f"Reconnection failed: {e}")

    def _publish_pose(self, result: Dict[str, Any], header: Header):
        """Pose 발행."""
        from scipy.spatial.transform import Rotation as R

        trans = result.get('translation', [0, 0, 0])
        rot_matrix = np.array(result.get('rotation_matrix', np.eye(3)))

        # Convert to quaternion
        quat = R.from_matrix(rot_matrix).as_quat()  # [x, y, z, w]

        # PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = self.frame_id
        pose_msg.pose.position.x = trans[0]
        pose_msg.pose.position.y = trans[1]
        pose_msg.pose.position.z = trans[2]
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

        self.pose_pub.publish(pose_msg)

        # PoseArray (for visualization in RViz)
        pose_array_msg = PoseArray()
        pose_array_msg.header = pose_msg.header
        pose_array_msg.poses.append(pose_msg.pose)
        self.pose_array_pub.publish(pose_array_msg)

        # TF broadcast
        if self.publish_tf:
            t = TransformStamped()
            t.header = pose_msg.header
            t.child_frame_id = self.object_frame_id
            t.transform.translation.x = trans[0]
            t.transform.translation.y = trans[1]
            t.transform.translation.z = trans[2]
            t.transform.rotation.x = quat[0]
            t.transform.rotation.y = quat[1]
            t.transform.rotation.z = quat[2]
            t.transform.rotation.w = quat[3]
            self.tf_broadcaster.sendTransform(t)

    def _publish_visualization(
        self,
        color: np.ndarray,
        result: Dict[str, Any],
        header: Header
    ):
        """시각화 이미지 발행."""
        vis = color.copy()

        trans = result.get('translation', [0, 0, 0])
        euler = result.get('euler_angles', {})
        latency = result.get('latency_ms', 0)

        # Text overlay
        texts = [
            f"X: {trans[0]*100:+6.2f} cm",
            f"Y: {trans[1]*100:+6.2f} cm",
            f"Z: {trans[2]*100:+6.2f} cm",
            f"Roll:  {euler.get('roll', 0):+7.2f}",
            f"Pitch: {euler.get('pitch', 0):+7.2f}",
            f"Yaw:   {euler.get('yaw', 0):+7.2f}",
            f"Latency: {latency:.0f}ms",
        ]

        y = 25
        for text in texts:
            cv2.putText(vis, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 25

        # Publish
        try:
            vis_msg = self.bridge.cv2_to_imgmsg(vis, 'bgr8')
            vis_msg.header = header
            self.vis_pub.publish(vis_msg)
        except CvBridgeError as e:
            rospy.logwarn(f"Visualization publish error: {e}")

    def _publish_status(self, status: str):
        """상태 메시지 발행."""
        self.status_pub.publish(String(data=status))

    def run(self):
        """노드 실행."""
        rospy.loginfo("FoundationPose ROS node running...")
        rospy.spin()

    def shutdown(self):
        """종료 처리."""
        rospy.loginfo("Shutting down...")
        self.socket.close()
        self.context.term()


def main():
    if not ROS_AVAILABLE:
        print("Error: ROS is not available.")
        print("Make sure to:")
        print("  1. Install ROS (Noetic/Melodic)")
        print("  2. Source setup.bash: source /opt/ros/noetic/setup.bash")
        print("  3. Install dependencies: pip install rospkg catkin_pkg")
        sys.exit(1)

    try:
        node = FoundationPoseROSNode()
        rospy.on_shutdown(node.shutdown)
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()

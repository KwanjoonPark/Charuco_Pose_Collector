import rosbag
import cv2
import cv2.aruco as aruco
import numpy as np
import csv
import argparse
import os
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from charuco_config import (
    CHARUCO_SQUARES_X as SQUARES_X,
    CHARUCO_SQUARES_Y as SQUARES_Y,
    CHARUCO_SQUARE_LENGTH as SQUARE_LENGTH,
    CHARUCO_MARKER_LENGTH as MARKER_LENGTH,
    ARUCO_DICT_TYPE,
    CHARUCO_OFFSET_X as OFFSET_X,
    CHARUCO_OFFSET_Y as OFFSET_Y,
    CHARUCO_OFFSET_Z as OFFSET_Z,
)

def get_offset_matrix(x, y, z):
    """
    Create a 4x4 transformation matrix with translation offset.

    Args:
        x, y, z: Translation offsets in meters

    Returns:
        4x4 homogeneous transformation matrix
    """
    mat = np.eye(4)
    mat[0, 3] = x
    mat[1, 3] = y
    mat[2, 3] = z
    return mat

def process_bag(bag_path, output_csv, show_video=True, save_images_on_success=False, frames_save_dir='frame_images', save_original=True, original_images_dir='images'):
    """
    Process ROS bag file to extract 6-DoF pose estimation data from ChArUco board.

    Args:
        bag_path: Path to the ROS bag file
        output_csv: Path to output CSV file
        show_video: Display video window during processing
        save_images_on_success: Save visualized frames where pose estimation succeeded
        frames_save_dir: Directory to save visualized frames
        save_original: Save original frames without overlay
        original_images_dir: Directory to save original frames
    """
    if save_images_on_success:
        if not os.path.exists(frames_save_dir):
            os.makedirs(frames_save_dir)
            print(f"Created directory for visualized frames: {frames_save_dir}")
    if save_original:
        if not os.path.exists(original_images_dir):
            os.makedirs(original_images_dir)
            print(f"Created directory for original images: {original_images_dir}")

    # ArUco dictionary setup (handle both old and new OpenCV APIs)
    use_new_api = hasattr(aruco, 'CharucoDetector')

    if use_new_api:
        # OpenCV 4.7+
        dictionary = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
        board = aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
        charuco_params = aruco.CharucoParameters()
        detector_params = aruco.DetectorParameters()
        detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        charuco_detector = aruco.CharucoDetector(board, charuco_params, detector_params)
    else:
        # Older OpenCV versions
        dictionary = aruco.Dictionary_get(ARUCO_DICT_TYPE)
        parameters = aruco.DetectorParameters_create()
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        board = aruco.CharucoBoard_create(SQUARES_X, SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH, dictionary)

    bag = rosbag.Bag(bag_path)
    bridge = CvBridge()

    color_image_topic = '/camera/color/image_raw'

    camera_info_topic = '/camera/color/camera_info'

    print("Analyzing bag file...")
    total_frames = bag.get_message_count(topic_filters=[color_image_topic])
    if total_frames == 0:
        print(f"Error: No image messages found in topic '{color_image_topic}'.")
        bag.close()
        return
    print(f"Found {total_frames} color image frames. Starting processing.")
    
    camera_matrix = None
    dist_coeffs = None

    
    with open(output_csv, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Timestamp', 'Image_File', 'Distance(cm)', 'Pitch(deg)', 'Yaw(deg)', 'Roll(deg)', 'X(cm)', 'Y(cm)', 'Z(cm)'])

        frame_count = 0
        success_count = 0
        fail_count = 0
        
        target_topics = [color_image_topic, camera_info_topic]

        for topic, msg, t in bag.read_messages(topics=target_topics):
            
            if topic == camera_info_topic:
                if camera_matrix is None:
                    camera_matrix = np.array(msg.K).reshape(3, 3)
                    dist_coeffs = np.array(msg.D)
                    print("✅ Camera information loaded successfully.")
                continue



            elif topic == color_image_topic:
                if camera_matrix is None:
                    print("Waiting for camera information...")
                    continue

                frame_count += 1
                try:
                    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
                except Exception as e:
                    print(f"Error converting color image: {e}")
                    continue

                if save_original:
                    original_image_filename = os.path.join(original_images_dir, f"frame_{frame_count:05d}.jpg")
                    cv2.imwrite(original_image_filename, cv_image)

                display_image = cv_image.copy() if show_video else cv_image
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

                pose_found = False
                marker_dist, yaw, pitch, roll = 0, 0, 0, 0
                final_t = [0, 0, 0]
                log_detail = "No markers detected."

                if use_new_api:
                    # OpenCV 4.7+ API
                    charuco_corners, charuco_ids, corners, ids = charuco_detector.detectBoard(gray)
                    if charuco_corners is not None and len(charuco_corners) >= 6:
                        log_detail = "ChArUco corners found but pose estimation failed."
                        obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
                        valid, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
                    else:
                        valid = False
                        if ids is not None and len(ids) > 0:
                            log_detail = "Markers detected but ChArUco interpolation failed."
                else:
                    # Older OpenCV API
                    corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=parameters)
                    valid = False
                    if ids is not None and len(ids) > 0:
                        log_detail = "Markers detected but ChArUco interpolation failed."
                        retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board)
                        if charuco_corners is not None and len(charuco_corners) >= 4:
                            log_detail = "ChArUco corners found but pose estimation failed."
                            valid, rvec, tvec = aruco.estimatePoseCharucoBoard(
                                charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None
                            )

                if valid:
                    pose_found = True

                    # Step 1: Convert marker coordinate system
                    R, _ = cv2.Rodrigues(rvec)
                    T_cam_marker = np.eye(4)
                    T_cam_marker[:3, :3] = R
                    T_cam_marker[:3, 3] = tvec.flatten()

                    # Step 2: Apply offset transformation
                    T_marker_target = get_offset_matrix(OFFSET_X, OFFSET_Y, OFFSET_Z)
                    T_cam_target = np.dot(T_cam_marker, T_marker_target)
                    final_t = T_cam_target[:3, 3]
                    x, y, z = final_t

                    # Step 3: Calculate distance from camera to target
                    marker_dist = np.sqrt(x**2 + y**2 + z**2)

                    # Step 4: Extract Euler angles from rotation matrix (ZYX convention)
                    # Using ZYX Euler angles (common convention in robotics)
                    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
                    singular = sy < 1e-6

                    if not singular:
                        # Non-singular case
                        rx = np.degrees(np.arctan2(R[2, 1], R[2, 2]))   # RX = Pitch
                        ry = np.degrees(np.arctan2(-R[2, 0], sy))      # RY = Yaw
                        rz = np.degrees(np.arctan2(R[1, 0], R[0, 0]))    # RZ = Roll
                    else:
                        # Gimbal lock case
                        rx = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
                        ry = np.degrees(np.arctan2(-R[2, 0], sy))
                        rz = 0

                    # Normalize pitch to be 0 when board faces camera directly
                    # RX is around ±180 when facing camera, normalize to [-90, 90]
                    # Forward tilt → negative, Backward tilt → positive
                    if rx > 90:
                        pitch = rx - 180
                    elif rx < -90:
                        pitch = rx + 180
                    else:
                        pitch = rx

                    yaw = ry
                    roll = rz

                    # Step 5: Project 3D points to 2D image for visualization
                    # Red dot: Final target point with offset applied
                    target_point_3d = np.array([final_t], dtype=np.float32)
                    target_pixel_2d, _ = cv2.projectPoints(target_point_3d, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
                    offset_target_pixel = (int(target_pixel_2d[0][0][0]), int(target_pixel_2d[0][0][1]))

                    # Blue dot: ChArUco board origin
                    charuco_origin_3d = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)
                    origin_pixel_2d, _ = cv2.projectPoints(charuco_origin_3d, rvec, tvec, camera_matrix, dist_coeffs)
                    charuco_origin_pixel = (int(origin_pixel_2d[0][0][0]), int(origin_pixel_2d[0][0][1]))

                    # Step 6: Visualization
                    if show_video:
                        # Draw axes manually (Z axis flipped)
                        axis_length = 0.05
                        axis_points = np.array([
                            [0, 0, 0],
                            [axis_length, 0, 0],    # X axis (red)
                            [0, axis_length, 0],    # Y axis (green)
                            [0, 0, -axis_length]    # Z axis (blue) - flipped
                        ], dtype=np.float32)
                        img_axis_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
                        pts = img_axis_points.reshape(-1, 2).astype(np.int32)
                        origin = tuple(pts[0])
                        x_end = tuple(pts[1])
                        y_end = tuple(pts[2])
                        z_end = tuple(pts[3])
                        cv2.line(display_image, origin, x_end, (0, 0, 255), 2)  # X: red
                        cv2.line(display_image, origin, y_end, (0, 255, 0), 2)  # Y: green
                        cv2.line(display_image, origin, z_end, (255, 0, 0), 2)  # Z: blue
                        # Draw 3D wireframe box (actual board dimensions)
                        WIDTH = SQUARES_X * SQUARE_LENGTH
                        HEIGHT = SQUARES_Y * SQUARE_LENGTH
                        DEPTH_VISIBLE = MARKER_LENGTH  # Arbitrary depth for visual representation of board thickness

                        # Define 8 corner vertices in board local coordinate system
                        # Origin (0,0,0) is the bottom-left corner of the board (aruco.estimatePoseCharucoBoard reference)
                        obj_points_box = np.array([
                            [0, 0, 0],
                            [WIDTH, 0, 0],
                            [WIDTH, HEIGHT, 0],
                            [0, HEIGHT, 0],
                            [0, 0, DEPTH_VISIBLE],
                            [WIDTH, 0, DEPTH_VISIBLE],
                            [WIDTH, HEIGHT, DEPTH_VISIBLE],
                            [0, HEIGHT, DEPTH_VISIBLE]
                        ], dtype=np.float32)

                        # Project 3D points to image plane
                        img_points_box, _ = cv2.projectPoints(obj_points_box, rvec, tvec, camera_matrix, dist_coeffs)
                        img_points_box = np.int32(img_points_box).reshape(-1, 2)

                        # Draw wireframe lines
                        color = (0, 255, 0)  # Green
                        thickness = 2

                        # Front face (Z=0)
                        cv2.line(display_image, tuple(img_points_box[0]), tuple(img_points_box[1]), color, thickness)
                        cv2.line(display_image, tuple(img_points_box[1]), tuple(img_points_box[2]), color, thickness)
                        cv2.line(display_image, tuple(img_points_box[2]), tuple(img_points_box[3]), color, thickness)
                        cv2.line(display_image, tuple(img_points_box[3]), tuple(img_points_box[0]), color, thickness)

                        # Back face (Z=DEPTH_VISIBLE)
                        cv2.line(display_image, tuple(img_points_box[4]), tuple(img_points_box[5]), color, thickness)
                        cv2.line(display_image, tuple(img_points_box[5]), tuple(img_points_box[6]), color, thickness)
                        cv2.line(display_image, tuple(img_points_box[6]), tuple(img_points_box[7]), color, thickness)
                        cv2.line(display_image, tuple(img_points_box[7]), tuple(img_points_box[4]), color, thickness)

                        # Connecting edges
                        cv2.line(display_image, tuple(img_points_box[0]), tuple(img_points_box[4]), color, thickness)
                        cv2.line(display_image, tuple(img_points_box[1]), tuple(img_points_box[5]), color, thickness)
                        cv2.line(display_image, tuple(img_points_box[2]), tuple(img_points_box[6]), color, thickness)
                        cv2.line(display_image, tuple(img_points_box[3]), tuple(img_points_box[7]), color, thickness)

                        cv2.putText(display_image, f"Distance: {marker_dist:.3f}m", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(display_image, f"Pitch: {pitch:.2f} deg", (10, 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(display_image, f"Yaw: {yaw:.2f} deg", (10, 90),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(display_image, f"Roll: {roll:.2f} deg", (10, 120),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(display_image, f"X: {x*100:.2f}cm", (10, 150),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(display_image, f"Y: {y*100:.2f}cm", (10, 180),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(display_image, f"Z: {z*100:.2f}cm", (10, 210),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                        # Blue dot: ChArUco board origin
                        cv2.circle(display_image, charuco_origin_pixel, 5, (255, 0, 0), -1)
                        # Red dot: Final target point with offset applied
                        cv2.circle(display_image, offset_target_pixel, 5, (0, 0, 255), -1)

                if pose_found:
                    success_count += 1
                    timestamp = f"{t.secs}.{t.nsecs}"

                    # Convert to cm for consistency
                    marker_dist_cm = marker_dist * 100  # meters to cm
                    x_cm = final_t[0] * 100
                    y_cm = final_t[1] * 100
                    z_cm = final_t[2] * 100

                    # Image filename format
                    image_file = f"frame_{frame_count:05d}.jpg"

                    csv_writer.writerow([
                        timestamp, image_file,
                        f"{marker_dist_cm:.2f}",
                        f"{pitch:.2f}", f"{yaw:.2f}", f"{roll:.2f}",
                        f"{x_cm:.2f}", f"{y_cm:.2f}", f"{z_cm:.2f}"
                    ])
                    print(f"[{frame_count}/{total_frames}] ✅ Success (Distance: {marker_dist_cm:.2f}cm, X: {x_cm:.2f}, Y: {y_cm:.2f}, Z: {z_cm:.2f})")

                    if save_images_on_success:
                        image_filename = os.path.join(frames_save_dir, f"frame_{frame_count:05d}.jpg")
                        cv2.imwrite(image_filename, display_image)
                else:
                    fail_count += 1
                    print(f"[{frame_count}/{total_frames}] ❌ Failed. Reason: {log_detail}")

                if show_video:
                    cv2.imshow("Pose Extraction", display_image)
                    if cv2.waitKey(1) == 27:
                        print("\nProcessing interrupted by user.")
                        break

    print("\n" + "="*30)
    print("    Processing Summary")
    print("="*30)
    print(f"Total frames processed: {frame_count}/{total_frames}")
    print(f"✅ Success: {success_count} frames")
    print(f"❌ Failed: {fail_count} frames")
    print(f"Results saved to: {output_csv}")
    print("="*30)

    bag.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract 6-DoF pose estimation data from ROS bag files using ChArUco board detection.'
    )
    parser.add_argument('--bag', type=str, default='/home/user/rosbags/raw/vcb_data_01.bag',
                        help='Path to the input ROS bag file')
    parser.add_argument('--out', type=str, default='/home/user/realsense_tools/scripts/6-DoF-Pose-Estimation/pose_data.csv',
                        help='Path to the output CSV file')
    parser.add_argument('--no-show', action='store_true',
                        help='Disable video display during processing')
    parser.add_argument('--save-images', action='store_true',
                        help='Save visualized frames where pose estimation succeeded')
    parser.add_argument('--frames-save-dir', type=str, default='frames',
                        help='Directory to save visualized frames (default: frames)')
    parser.add_argument('--save-original', action='store_true',
                        help='Save original images without overlay')
    parser.add_argument('--original-images-dir', type=str, default='images',
                        help='Directory to save original images (default: images)')
    args = parser.parse_args()

    process_bag(args.bag, args.out, not args.no_show, args.save_images, args.frames_save_dir, args.save_original, args.original_images_dir)
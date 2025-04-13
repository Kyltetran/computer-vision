import cv2 as cv
import numpy as np
import open3d as o3d
import os
import glob
import random

#FIXME: calibrate camera

##############################
# Calibration from Image Folders
##############################
def calibrate_from_folders(left_folder, right_folder, board_size=(8, 6), square_size=1.0):
    """
    Load exactly 6 calibration images from each folder and use them to calibrate both cameras.
    Assumes that each folder contains calibration images of a chessboard of specified board_size.

    Args:
      left_folder: Path to calibration images from the left camera.
      right_folder: Path to calibration images from the right camera.
      board_size: Tuple (columns, rows) for the number of inner corners on the chessboard.
      square_size: Real-world size of a square (in your preferred units).

    Returns:
      leftMapX, leftMapY, rightMapX, rightMapY, Q, imageSize: Rectification maps, reprojection matrix,
      and image size (width, height).
    """
    left_images = sorted(glob.glob(os.path.join(left_folder, "*.png")))
    right_images = sorted(glob.glob(os.path.join(right_folder, "*.png")))

    if len(left_images) < 6 or len(right_images) < 6:
        raise Exception("Not enough calibration images in one or both folders (need at least 6 each).")

    # Select the first 6 images from each folder.
    left_images = left_images[:6]
    right_images = right_images[:6]

    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Scale the object points

    objpoints = []  # 3D real-world points
    imgpoints_left = []  # 2D points in left images
    imgpoints_right = []  # 2D points in right images
    imageSize = None

    # Process each pair of images:
    for l_file, r_file in zip(left_images, right_images):
        imgL = cv.imread(l_file)
        imgR = cv.imread(r_file)
        if imgL is None or imgR is None:
            print(f"Warning: Could not load {l_file} or {r_file}; skipping this pair.")
            continue

        if imageSize is None:
            imageSize = (imgL.shape[1], imgL.shape[0])  # width, height

        grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

        retL, cornersL = cv.findChessboardCorners(grayL, board_size, None)
        retR, cornersR = cv.findChessboardCorners(grayR, board_size, None)
        if retL and retR:
            criteria = (cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER, 30, 0.001)
            cornersL = cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            cornersR = cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints_left.append(cornersL)
            imgpoints_right.append(cornersR)
            print(f"Chessboard detected in pair: {l_file} & {r_file}")
        else:
            print(f"Chessboard not detected in pair: {l_file} & {r_file}")

    if len(objpoints) < 1:
        raise Exception("No valid calibration pairs found.")

    # Calibrate each camera individually:
    retL, mtxL, distL, _, _ = cv.calibrateCamera(objpoints, imgpoints_left, imageSize, None, None)
    retR, mtxR, distR, _, _ = cv.calibrateCamera(objpoints, imgpoints_right, imageSize, None, None)
    if not retL or not retR:
        raise Exception("Individual camera calibration failed.")
    print("Individual camera calibration completed.")

    # Stereo calibration (fixing intrinsics):
    flags = cv.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv.TermCriteria_COUNT + cv.TermCriteria_EPS, 100, 1e-5)
    retStereo, mtxL, distL, mtxR, distR, R, T, E, F = cv.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtxL, distL, mtxR, distR,
        imageSize, criteria=criteria_stereo, flags=flags
    )
    if not retStereo:
        raise Exception("Stereo calibration failed.")

    # Stereo rectification:
    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(mtxL, distL, mtxR, distR, imageSize, R, T, alpha=0)

    leftMapX, leftMapY = cv.initUndistortRectifyMap(mtxL, distL, R1, P1, imageSize, cv.CV_16SC2)
    rightMapX, rightMapY = cv.initUndistortRectifyMap(mtxR, distR, R2, P2, imageSize, cv.CV_16SC2)

    print("Stereo calibration and rectification completed from folder images.")
    return leftMapX, leftMapY, rightMapX, rightMapY, Q, imageSize


##############################
# Stereo Depth Functions with Cleaner Output
##############################
def compute_stereo_depth(grayL, grayR):
    """
    Compute a depth (disparity) map using StereoSGBM, with parameters tuned
    for a smoother, less noisy depth output.
    """
    window_size = 5  # block size
    min_disp = 0
    num_disp = 16 * 5  # must be divisible by 16; adjust as needed
    stereo = cv.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    # Normalize the disparity (depth map) to 0-255, then smooth it.
    depth = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX)
    depth = np.uint8(depth)
    depth = cv.medianBlur(depth, 5)
    depth = cv.bilateralFilter(depth, 9, 75, 75)
    return depth


def compute_depth_from_matches(imgL, imgR):
    """Compute depth map using feature matching combined with a stereo matching fallback."""
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(grayL, None)
    kp2, des2 = sift.detectAndCompute(grayR, None)

    if des1 is None or des2 is None or len(des1) < 8 or len(des2) < 8:
        print("Not enough features found, falling back to stereo matching only")
        return compute_stereo_depth(grayL, grayR)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except Exception as e:
        print(f"Error in feature matching: {e}")
        return compute_stereo_depth(grayL, grayR)

    good_matches = []
    pts1 = []
    pts2 = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    if len(good_matches) < 8:
        print("Not enough good matches found, falling back to stereo matching only")
        return compute_stereo_depth(grayL, grayR)

    pts1 = np.float32(pts1).reshape(-1, 1, 2)
    pts2 = np.float32(pts2).reshape(-1, 1, 2)

    try:
        F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC, 3.0, 0.99)
        if F is None or F.shape != (3, 3):
            print("Could not compute fundamental matrix, falling back to stereo matching only")
            return compute_stereo_depth(grayL, grayR)

        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]
        height, width = grayL.shape
        focal = width
        K = np.array([[focal, 0, width / 2],
                      [0, focal, height / 2],
                      [0, 0, 1]])
        E = K.T @ F @ K
        _, R, t, mask_pose = cv.recoverPose(E, pts1, pts2, K)
        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K @ np.hstack((R, t))
        pts1_norm = cv.undistortPoints(pts1, K, None)
        pts2_norm = cv.undistortPoints(pts2, K, None)
        points_4d = cv.triangulatePoints(P1, P2, pts1_norm, pts2_norm)
        points_3d = points_4d[:3] / points_4d[3]
        depth_map = np.zeros_like(grayL, dtype=np.float32)
        for i, pt in enumerate(pts1):
            x, y = pt.ravel()
            if 0 <= x < width and 0 <= y < height:
                depth_map[int(y), int(x)] = abs(points_3d[2][i])
        stereo_depth = compute_stereo_depth(grayL, grayR)
        mask_depth = depth_map > 0
        combined_depth = stereo_depth.copy()
        combined_depth[mask_depth] = depth_map[mask_depth]
        combined_depth = cv.bilateralFilter(combined_depth, 9, 75, 75)
        return combined_depth
    except Exception as e:
        print(f"Error in depth computation: {e}")
        return compute_stereo_depth(grayL, grayR)


##############################
# Point Cloud Generation (unchanged)
##############################
def generate_point_cloud(disparity, color_image, Q, output_file='output/point_cloud.ply'):
    """Generate point cloud and save to PLY file in ASCII format"""
    points_3D = cv.reprojectImageTo3D(disparity, Q)
    colors = cv.cvtColor(color_image, cv.COLOR_BGR2RGB)
    mask = disparity > disparity.min()
    output_points = points_3D[mask]
    output_colors = colors[mask]
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(output_points)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for p, c in zip(output_points, output_colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")
    print(f"Point cloud saved to {output_file}")


##############################
# Main Process
##############################
def main():
    # --- Calibration from folders ---
    try:
        leftMapX, leftMapY, rightMapX, rightMapY, Q, imageSize = calibrate_from_folders("left_cam", "right_cam",
                                                                                        board_size=(8, 6))
        use_calibration = True
        print("Calibration from folders completed successfully.")
    except Exception as e:
        print(f"Error during calibration: {e}")
        print("Proceeding without calibration.")
        use_calibration = False
        width, height = 640, 480
        imageSize = (width, height)
        Q = np.float32([[1, 0, 0, -width / 2.0],
                        [0, 1, 0, -height / 2.0],
                        [0, 0, 0, 1000],
                        [0, 0, -1 / 0.1, 0]])

    # --- Open stereo video files ---
    capL = cv.VideoCapture('left.mp4')
    capR = cv.VideoCapture('right.mp4')

    if not capL.isOpened() or not capR.isOpened():
        print("Error opening video streams.")
        return

    chosen_frame = None
    # Process video streams (display video while capturing) and store one frame pair.
    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            break

        if use_calibration:
            frameL = cv.remap(frameL, leftMapX, leftMapY, cv.INTER_LINEAR)
            frameR = cv.remap(frameR, rightMapX, rightMapY, cv.INTER_LINEAR)

        cv.imshow("Left", frameL)
        cv.imshow("Right", frameR)

        chosen_frame = (frameL, frameR)  # always update; at the end, the last valid frame will be used.

        if cv.waitKey(1) == 27:  # exit on ESC
            break

    capL.release()
    capR.release()
    cv.destroyAllWindows()

    if chosen_frame is None:
        print("No valid frame was extracted from the video.")
        return

    final_frameL, final_frameR = chosen_frame
    # Compute depth map using combined matching (with SGBM fallback for cleaner output)
    depth_map = compute_depth_from_matches(final_frameL, final_frameR)
    # Save depth map in output folder (using a cleaner, normalized version)
    os.makedirs("output", exist_ok=True)
    depth_map_vis = cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX)
    depth_map_vis = np.uint8(depth_map_vis)
    depth_output_file = os.path.join("output", "depth_map.png")
    cv.imwrite(depth_output_file, depth_map_vis)
    print(f"Depth map saved as '{depth_output_file}'.")

    # Generate point cloud from the same frame
    point_cloud_file = os.path.join("output", "point_cloud.ply")
    generate_point_cloud(depth_map, final_frameL, Q, output_file=point_cloud_file)


if __name__ == "__main__":
    main()

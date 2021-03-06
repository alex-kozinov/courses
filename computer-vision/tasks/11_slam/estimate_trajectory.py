import os
import pickle
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from copy import deepcopy

from common.dataset import Dataset
from common.trajectory import Trajectory
from common.intrinsics import Intrinsics


def draw_matches(data_dir, img1_path_r, img2_path_r, kp1, kp2, matches, ax):
    img1 = cv.imread(os.path.join(data_dir, img1_path_r), 0)
    img2 = cv.imread(os.path.join(data_dir, img2_path_r), 0)

    matches = [[m] for m in matches]
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=2)
    ax.imshow(img3, )


def rvec_tvec_to_matrix(rvec, tvec):
    rot_matrix, _ = cv.Rodrigues(rvec)
    res = np.zeros((4, 4))
    res[:3, :3] = np.linalg.inv(rot_matrix)
    res[:3, 3] = tvec.ravel()
    res[3, 3] = 1
    return res

def quaternion_to_rotation_matrix(quaternion):
    """
    Generate rotation matrix 3x3  from the unit quaternion.

    Input:
    qQuaternion -- tuple consisting of (qx,qy,qz,qw) where
         (qx,qy,qz,qw) is the unit quaternion.

    Output:
    matrix -- 3x3 rotation matrix
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    eps = np.finfo(float).eps * 4.0
    assert nq > eps
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3]),
        (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3]),
        (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1])
    ), dtype=np.float64)


def triangulate_projection_score(img1_points, img2_points, img1_pos, img2_pos, intrinsics):
    """

    Input:
    img1_points -- first image coordinates Nx2 ndarray
    img2_points -- second image coordinates Nx2 ndarray
    Output:
    """
    tq = Trajectory.from_matrix4(img1_pos)
    img1_translation = np.array(tq[:3], ndmin=2)
    img1_quaternion = np.array(tq[3:])
    tq = Trajectory.from_matrix4(img2_pos)
    img2_translation = np.array(tq[:3], ndmin=2)
    img2_quaternion = np.array(tq[3:])

    # compute rotation matrix from the quaternion
    # here we invert the rotation matrix because we need an inverse transfromation
    img1_rotation = np.linalg.inv(quaternion_to_rotation_matrix(img1_quaternion))
    # compute Rodrigues vector from rotation matrix (is needed for OpenCV)
    img1_rodrigues, _ = cv.Rodrigues(img1_rotation)
    # update translation as well for the inverse transformation
    img1_translation = -1.0 * np.matmul(img1_rotation, img1_translation.T)

    # the same for the second image
    img2_rotation = np.linalg.inv(quaternion_to_rotation_matrix(img2_quaternion))
    img2_rodrigues, _ = cv.Rodrigues(img2_rotation)
    img2_translation = -1.0 * np.matmul(img2_rotation, img2_translation.T)

    # define the intrinsic matrix
    K = np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0]
        ]
    )

    # compute projection matrices
    img1_projection_matrix = np.matmul(K, np.concatenate((img1_rotation, img1_translation), axis=1))
    img2_projection_matrix = np.matmul(K, np.concatenate((img2_rotation, img2_translation), axis=1))

    # triangulate 3D point
    points_4d = cv.triangulatePoints(
        img1_projection_matrix,
        img2_projection_matrix,
        img1_points.T,
        img2_points.T
    )
    points_3d = 1. * points_4d[0:3, :] / points_4d[3, :]

    # reproject 3D point back to the images
    img1_point_reprojection, _ = cv.projectPoints(
        points_3d.T,
        img1_rodrigues,
        img1_translation,
        K,
        None
    )
    img1_point_reprojection = img1_point_reprojection.reshape(img1_point_reprojection.shape[0], 2)
    reprojection_error_1 = np.linalg.norm(img1_points - img1_point_reprojection, axis=1)

    img2_point_reprojection, _ = cv.projectPoints(
        points_3d.T,
        img2_rodrigues,
        img2_translation,
        K,
        None
    )
    img2_point_reprojection = img1_point_reprojection.reshape(img2_point_reprojection.shape[0], 2)
    reprojection_error_2 = np.linalg.norm(img2_points - img2_point_reprojection, axis=1)
    return points_3d.T, (reprojection_error_1 + reprojection_error_2) / 2


def get_most_relevant_matches(matches, n, m, max_matches=None):
    matches = sorted(matches, key=lambda x: x.distance)
    checked_1 = [0 for _ in range(n)]
    checked_2 = [0 for _ in range(m)]
    checked_matches = []

    for match in matches:
        if checked_1[match.queryIdx] or checked_2[match.trainIdx]:
            continue

        checked_1[match.queryIdx] = 1
        checked_2[match.trainIdx] = 1

        if max_matches and len(checked_matches) == max_matches:
            break
        checked_matches.append(match)

    return checked_matches


def get_pairwise_matches(kp1, kp2, des1, des2, max_matches=None):
    pairwise_matches = []

    # FLANN_INDEX_LSH = 6
    # index_params = dict(algorithm=FLANN_INDEX_LSH,
    #                     table_number=6,  # 12
    #                     key_size=12,  # 20
    #                     multi_probe_level=1)  # 2
    # flann = cv.FlannBasedMatcher(index_params)
    # matches = flann.knnMatch(des1, des2, k=2)
    # FLANN parameters
    des1 = np.array(des1)
    des2 = np.array(des2)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for i, m_n in enumerate(matches):
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)

    # for m in good_matches:
    #     if m.queryIdx >= len(kp1):
    #         print('w')
    if not len(good_matches):
        return []
    pts1 = np.int32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.int32([kp2[m.trainIdx].pt for m in good_matches])
    # try:
    _, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)
    # except:
    #     # print("WOW")
    #     assert False

    if mask is None:
        return []
    for i, good_match in enumerate(good_matches):
        if mask[i] == 1:
            pairwise_matches.append(good_match)

    pairwise_matches = get_most_relevant_matches(
        pairwise_matches,
        len(kp1),
        len(kp2),
        max_matches=max_matches
    )

    return pairwise_matches


def triangulation(kp1, kp2, T_1w, T_2w):
    """Triangulation to get 3D points
    Args:
        kp1: list of keypoints in view 1 (normalized)
        kp2: list of keypoints in view 2 (normalized)
        T_1w (4x4): pose of view 1 w.r.t  i.e. T_1w (from w to 1)
        T_2w (4x4): pose of view 2 w.r.t world, i.e. T_2w (from w to 2)
    Returns:
        X (Nx4): list of 4D coordinates of the keypoints w.r.t world coordinate
    """
    kp1 = np.array([kp.pt for kp in kp1])
    kp2 = np.array([kp.pt for kp in kp2])
    X = cv.triangulatePoints(T_1w[:3], T_2w[:3], kp1.T, kp2.T)
    return X[:3, ].T


def compute_camera_pos(object_points, image_points, intrinsics):
    K = np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0]
        ]
    )
    try:
        _, rvec, tvec, inliers = cv.solvePnPRansac(object_points, image_points, K, None)
    # _, rvec, tvec = cv.solvePnP(object_points, image_points, K, None)
    except:
        print ('sad')
        # print(object_points, image_points)
        # _, rvec, tvec, inliers = cv.solvePnPRansac(object_points, image_points, K, None)
        _, rvec, tvec = cv.solvePnP(object_points, image_points, K, None)
        # assert False
    return rvec, tvec

def resolve_triple_match(matches_1, matches_2):
    """
    find correct matches from two references images X_1, X_2 to test image Y
    Args:
        matches_1: match objects withe query image X_1 and train Y
        matches_2: match objects withe query image X_2 and train Y
    Returns:
        x_indexes (N, ): list with correct indexes of key points for X_1, X_2 w.r.t Y
        y_indexes (N, ): list with correct indexes of key points for Y with correct patching to X_1, X_2
    """
    matches_pairs = {}
    x_indexes = []
    y_indexes = []
    for match_group in [matches_1, matches_2]:
        for match in match_group:
            x = match.queryIdx
            matches_pairs.setdefault(x, [])
            matches_pairs[x].append(match)

    for x, match_pair in matches_pairs.items():
        if len(match_pair) < 2:
            continue
        if match_pair[0].trainIdx != match_pair[1].trainIdx:
            continue
        x_indexes.append(x)
        y_indexes.append(match_pair[0].trainIdx)
    return x_indexes, y_indexes


class Matcher:
    def __init__(self, intrinsics):
        self.n = 0
        self.kps = None
        self.descriptors = None
        self.poses = None
        self.intrinsics = intrinsics
        self.matches = {}
        self.matches_poses = {}
        self.edges = []
        self.vertexes = []
        self.union_set = {}
        self.vertex_poses = {}
        self.union_poses = {}
        self.is_union_valid = {}

    def fit_pairwise_matches(self, kps, descriptors, poses, projection_threshold, max_matches=None):
        self.n = len(kps)
        self.kps = kps
        self.descriptors = descriptors
        self.poses = poses

        # print('Start fitting pairwise matches')
        for j in range(self.n):
            # print(f'{j}/{self.n - 1}')
            for i in range(j):
                self.matches[(i, j)] = get_pairwise_matches(
                    self.kps[i],
                    self.kps[j],
                    self.descriptors[i],
                    self.descriptors[j],
                    max_matches
                )
                matches = self.matches[(i, j)]
                img1_points = np.float64([self.kps[i][m.queryIdx].pt for m in matches])
                img2_points = np.float64([self.kps[j][m.trainIdx].pt for m in matches])
                points_3d, scores = triangulate_projection_score(
                    img1_points,
                    img2_points,
                    self.poses[i],
                    self.poses[j],
                    self.intrinsics
                )
                acceptance_mask = scores <= projection_threshold
                good_matches = []
                matches_poses = []
                for match_idx, match in enumerate(matches):
                    if acceptance_mask[match_idx]:
                        good_matches.append(match)
                        matches_poses.append(points_3d[match_idx])
                self.matches[(i, j)] = good_matches
                self.matches_poses[(i, j)] = matches_poses

    def get_pairwise_matches_by_ids(self, ind1, ind2):
        curr_matches = self.matches.get((ind1, ind2))
        return curr_matches

    def get_matched_points_info_by_ids(self, ind1, ind2):
        matches = self.get_pairwise_matches_by_ids(ind1, ind2)
        kps1 = []
        kps2 = []
        des1 = []
        des2 = []
        for m in matches:
            x = m.queryIdx
            y = m.trainIdx
            kps1.append(self.kps[ind1][x])
            kps2.append(self.kps[ind2][y])
            des1.append(self.descriptors[ind1][x])
            des2.append(self.descriptors[ind2][y])
        return kps1, des1, kps2, des2, self.matches_poses[(ind1, ind2)]


class SLAM:
    def __init__(self, data_dir, cache_dir, intrinsics):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.intrinsics = intrinsics
        self.train_images = []
        self.train_poses = []
        self.train_kp = []
        self.train_des = []
        self.max_matches = None

        self.key_point_descriptors_cached_file_ = os.path.join(cache_dir, 'key_point_descriptors.pkl')
        self.key_point_descriptors_cached_ = {}
        self.matcher = Matcher(intrinsics)
        self.load_data()

    def load_data(self):
        if os.path.exists(self.key_point_descriptors_cached_file_):
            with open(self.key_point_descriptors_cached_file_, 'rb') as f:
                pass
                # self.key_point_descriptors_cached_ = pickle.load(f)

    def dump_data(self):
        os.makedirs(os.path.dirname(self.key_point_descriptors_cached_file_), exist_ok=True)
        with open(self.key_point_descriptors_cached_file_, 'wb') as f:
            pass

    def fit(self, train_images, train_poses, projection_threshold=10, max_matches=None):
        self.train_images = train_images
        self.max_matches = max_matches
        train_kp, train_des = self.get_lists_key_point_descriptors(train_images)
        self.train_poses = train_poses
        self.matcher.fit_pairwise_matches(train_kp, train_des, train_poses, projection_threshold, max_matches=max_matches)

    def predict(self, test_images):
        test_kps, test_deses = self.get_lists_key_point_descriptors(test_images)
        res = []
        # print("computing poses for test images")
        for k, (test_kp, test_des) in enumerate(zip(test_kps, test_deses)):
            appropriate_pairs = []
            # print(f'{k}/{len(test_deses)}')
            for j in range(len(self.train_images)):
                for i in range(j):
                    kps1, des1, kps2, des2, points_3d = self.matcher.get_matched_points_info_by_ids(i, j)

                    matches_1 = get_pairwise_matches(kps1, test_kp, des1, test_des, self.max_matches)
                    matches_2 = get_pairwise_matches(kps2, test_kp, des2, test_des, self.max_matches)

                    train_good_indexes, test_good_indexes = resolve_triple_match(matches_1, matches_2)
                    assert len(train_good_indexes) == len(test_good_indexes)
                    if len(test_good_indexes) < 6:
                        continue

                    object_points = np.array(points_3d)[train_good_indexes]
                    image_points = np.array([test_kp[idx].pt for idx in test_good_indexes])
                    train_1_image_points = np.array([kps1[idx].pt for idx in train_good_indexes])
                    train_2_image_points = np.array([kps2[idx].pt for idx in train_good_indexes])
                    # dist = np.linalg.norm(train_1_image_points - train_2_image_points)
                    dist = Trajectory.compute_distance(self.train_poses[i] - self.train_poses[j])

                    appropriate_pairs.append([
                        dist,
                        (i, j),
                        object_points,
                        image_points,
                        train_good_indexes,
                        test_good_indexes,
                    ])

            if not len(appropriate_pairs):
                res.append(self.train_poses[0])
                print("Fall back")
                continue

            # appropriate_pairs = sorted(appropriate_pairs, key=lambda x: len(x[-2]))
            appropriate_pairs = sorted(appropriate_pairs, key=lambda x: x[0])

            _, (i, j), object_points, image_points, train_indexes, _ = appropriate_pairs[-1]
            rvec, tvec = compute_camera_pos(object_points, image_points, self.intrinsics)

            # rvecs, tvecs = [], []
            # appropriate_pairs = appropriate_pairs[len(appropriate_pairs) // 2:]
            # for _, (i, j), object_points, image_points, train_indexes, _ in appropriate_pairs:
            #     rvec_curr, tvec_curr = compute_camera_pos(object_points, image_points, self.intrinsics)
            #     rvecs.append(rvec_curr)
            #     tvecs.append(tvec_curr)
            # rvec = np.mean(rvecs, axis=0)
            # tvec = np.mean(tvecs, axis=0)

            matrix = rvec_tvec_to_matrix(rvec, tvec)
            res.append(matrix)

            # for pair_ind, (_, (i, j), _, _, train_indexes, test_indexes) in enumerate(appropriate_pairs):
            #     if pair_ind != 0 and pair_ind != (len(appropriate_pairs) - 1):
            #         continue
            #     matches = self.matcher.get_pairwise_matches_by_ids(i, j)
            #     kp1 = self.matcher.kps[i]
            #     kp2 = self.matcher.kps[j]
            #     good_test_kp = np.array(test_kp)[test_indexes]
            #     matches = np.array(matches)[train_indexes]
            #
            #     title = 'minimal distance' if (pair_ind == 0) else 'maximal distance'
            #     plt.figure(figsize=(24, 30))
            #     ax1 = plt.subplot(211)
            #     ax1.set_title(title, fontsize=25)
            #     draw_matches(self.data_dir, self.train_images[i], self.train_images[j], kp1, kp2, matches, ax1)
            #
            #     ax2 = plt.subplot(212)
            #     test_img = cv.imread(os.path.join(self.data_dir, test_images[k]), 0)
            #     test_img = cv.drawKeypoints(test_img, good_test_kp, None, color=(0, 255, 0), flags=0)
            #     ax2.imshow(test_img, )
            #     plt.show()



        return res

    def get_lists_key_point_descriptors(self, img_paths):
        kp_list = []
        des_list = []
        for img_path_r in img_paths:
            full_img_path = os.path.join(self.data_dir, img_path_r)
            kp, des = self.get_single_key_point_descriptors(full_img_path)
            kp_list.append(kp)
            des_list.append(des)
            # break
        return kp_list, des_list

    def get_single_key_point_descriptors(self, img_path, method='sift'):
        kp_des = self.key_point_descriptors_cached_.get(img_path)
        if kp_des is not None:
            kp, des = kp_des
            return kp, des

        img = cv.imread(img_path, 0)
        if method == 'orb':
            # Initiate ORB detector
            orb = cv.ORB_create()
            # sift = cv.SIFT()  # try to use it
            # find the keypoints with ORB
            kp = orb.detect(img, None)
            # compute the descriptors with ORB
            kp, des = orb.compute(img, kp)
        elif method == 'sift':
            # Initiate SIFT detector
            sift = cv.SIFT_create()
            # find the keypoints and descriptors with SIFT
            kp = sift.detect(img, None)
            # compute the descriptors with ORB
            kp, des = sift.compute(img, kp)

        # img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
        # plt.imshow(img2), plt.show()
        self.key_point_descriptors_cached_[img_path] = (kp, des)
        return kp, des


def estimate_trajectory(data_dir, out_dir):
    # TODO: fill trajectory here
    all_images = Dataset.read_lists(Dataset.get_rgb_list_file(data_dir), value_type=str)
    train_poses = Dataset.read_lists(Dataset.get_known_poses_file(data_dir), value_type=float)
    intrinsics = Intrinsics.read(Dataset.get_intrinsics_file(data_dir))
    train_labels = []

    for i, pose in enumerate(train_poses):
        # if i > 10:
        #     break
        pose[0] = int(pose[0])
        train_labels.append(pose[0])

    train_images = [image for image in all_images if int(image[0]) in train_labels]
    test_images = [image for image in all_images if int(image[0]) not in train_labels]

    train_images_paths = [image[1] for image in train_images]
    train_images_poses = [Trajectory.to_matrix4(pose[1:]) for pose in train_poses]
    test_images_paths = [image[1] for image in test_images]

    slam = SLAM(data_dir, os.path.join(data_dir, 'cache'), intrinsics)
    slam.fit(train_images_paths, train_images_poses, 10, 800)

    test_images_poses = slam.predict(test_images_paths)
    trajectory = {}
    for i, image_pos in enumerate(train_images_poses):
        label = int(train_poses[i][0])
        trajectory[label] = image_pos
    for i, image_pos in enumerate(test_images_poses):
        label = int(test_images[i][0])
        trajectory[label] = image_pos

    Trajectory.write(Dataset.get_result_poses_file(out_dir), trajectory)
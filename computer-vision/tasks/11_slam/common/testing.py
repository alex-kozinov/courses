import os
import argparse

from common.dataset import Dataset
from common.trajectory import Trajectory
from common.absolute_translational_error import AbsoluteTranslationalError
from common.relative_pose_error import RelativePoseError


INLIERS_THRESHOLD = 0.7
DISTANCE_THRESHOLD = 0.2  # m
ANGLE_THRESHOLD = 0.25  # radians, about 15 degrees

GOOD_ENOUGH_INLIERS_COUNT = 0.99
GOOD_ENOUGH_DISTANCE = 0.015  # m
GOOD_ENOUGH_ANGLE = 0.02  # radians, about 1 degrees


def read_ids_to_ignore(data_path):
    return list(Trajectory.read(Dataset.get_known_poses_file(data_path)).keys())


def compute_inliers_score(test_trajectory, gt_trajectory, all_tests):
    inliers = []
    for key, test in test_trajectory.items():
        gt = gt_trajectory[key]
        diff = Trajectory.compute_relative_transform(gt, test)
        if Trajectory.compute_angle(diff) < ANGLE_THRESHOLD and Trajectory.compute_distance(diff) < DISTANCE_THRESHOLD:
            inliers.append(key)

    return float(len(inliers)) / all_tests, inliers


def test_result(test_data_path, private_data_path, result_filename, test_inliers_only=False):
    test_trajectory = Trajectory.read(Dataset.get_result_poses_file(test_data_path))
    gt_trajectory = Trajectory.read(Dataset.get_ground_truth_file(private_data_path))
    ignored_ids = read_ids_to_ignore(private_data_path)

    unknown_poses_num = len(gt_trajectory) - len(ignored_ids)

    test_trajectory = {key: value for key, value in test_trajectory.items() if key not in ignored_ids}
    gt_trajectory = {key: value for key, value in gt_trajectory.items() if key in test_trajectory}

    inliers_score, inliers = compute_inliers_score(test_trajectory, gt_trajectory, unknown_poses_num)

    if test_inliers_only:
        test_trajectory = {key: value for key, value in test_trajectory.items() if key in inliers}
        gt_trajectory = {key: value for key, value in gt_trajectory.items() if key in test_trajectory}

    ate_error = AbsoluteTranslationalError.estimate(test_trajectory, gt_trajectory)
    rpe_error = RelativePoseError.estimate(test_trajectory, gt_trajectory)

    with open(result_filename, 'w') as file:
        file.write('inliers                      {}\n'.format(inliers_score))

        file.write('absolute.translation.rmse    {}\n'.format(ate_error.rmse))
        file.write('absolute.translation.mean    {}\n'.format(ate_error.mean))
        file.write('absolute.translation.median  {}\n'.format(ate_error.median))
        file.write('absolute.translation.std     {}\n'.format(ate_error.std))
        file.write('absolute.translation.min     {}\n'.format(ate_error.min))
        file.write('absolute.translation.max     {}\n'.format(ate_error.max))

        file.write('relative.translation.rmse    {}\n'.format(rpe_error.translation_rmse))
        file.write('relative.translation.mean    {}\n'.format(rpe_error.translation_mean))
        file.write('relative.translation.median  {}\n'.format(rpe_error.translation_median))
        file.write('relative.translation.std     {}\n'.format(rpe_error.translation_std))
        file.write('relative.translation.min     {}\n'.format(rpe_error.translation_min))
        file.write('relative.translation.max     {}\n'.format(rpe_error.translation_max))
        file.write('relative.rotation.rmse       {}\n'.format(rpe_error.rotation_rmse))
        file.write('relative.rotation.mean       {}\n'.format(rpe_error.rotation_mean))
        file.write('relative.rotation.median     {}\n'.format(rpe_error.rotation_median))
        file.write('relative.rotation.std        {}\n'.format(rpe_error.rotation_std))
        file.write('relative.rotation.min        {}\n'.format(rpe_error.rotation_min))
        file.write('relative.rotation.max        {}\n'.format(rpe_error.rotation_max))

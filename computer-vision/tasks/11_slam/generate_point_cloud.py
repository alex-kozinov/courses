#!/usr/bin/python3
import argparse

from common.dataset import Dataset
from common.trajectory import Trajectory
from common.point_cloud import PointCloud


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('public', help='Path to public part of a dataset')
    parser.add_argument('private', help='Path to private part of a dataset')
    parser.add_argument('trajectory', help='Path to file with trajectory of the frames')
    parser.add_argument('output', help='Output point cloud in .ply format')
    args = parser.parse_args()

    trajectory = Trajectory.read(args.trajectory)

    points, colors = PointCloud.generate_dataset_cloud(args.public, args.private, trajectory, 8)
    PointCloud.write_ply(args.output, points, colors)


if __name__ == '__main__':
    main()

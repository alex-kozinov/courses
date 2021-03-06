import os
import numpy
import skimage.io

from .dataset import Dataset
from .intrinsics import Intrinsics


class PointCloud:
    @staticmethod
    def generate_frame_cloud(rgb_file, depth_file, intrinsics, transform, downsampling_step=1, depth_scale=5000):
        rgb = skimage.io.imread(rgb_file)
        depth = skimage.io.imread(depth_file)

        assert rgb.shape[0:2] == depth.shape, 'Color and depth image do not have the same resolution.'
        assert rgb.dtype == 'uint8', 'Color image is not in RGB format'
        assert depth.dtype == 'uint16', 'Depth image is not in intensity format'

        points = []
        colors = []
        for v in range(0, rgb.shape[0], downsampling_step):
            for u in range(0, rgb.shape[1], downsampling_step):
                color = rgb[v, u]
                Z = depth[v, u] / depth_scale
                if Z == 0:
                    continue
                X = (u - intrinsics.cx) * Z / intrinsics.fx
                Y = (v - intrinsics.cy) * Z / intrinsics.fy
                vec_org = numpy.matrix([[X], [Y], [Z], [1]])
                vec_transf = numpy.dot(transform, vec_org)
                points.append((vec_transf[0, 0], vec_transf[1, 0], vec_transf[2, 0]))
                colors.append(color)

        return points, colors

    @staticmethod
    def generate_dataset_cloud(public_data_path, private_data_path, trajectory, downsampling_step=1, depth_scale=5000):
        rgb_list = Dataset.read_dict_of_lists(Dataset.get_rgb_list_file(public_data_path))
        depth_list = Dataset.read_dict_of_lists(Dataset.get_depth_list_file(private_data_path))
        assert len(rgb_list) == len(depth_list)
        assert len(rgb_list) == len(trajectory)

        intrinsics = Intrinsics.read(Dataset.get_intrinsics_file(public_data_path))

        all_points = []
        all_colors = []
        for frame_id, rgb_file in rgb_list.items():
            points, colors = PointCloud.generate_frame_cloud(
                os.path.join(public_data_path, rgb_file),
                os.path.join(private_data_path, depth_list[frame_id]),
                intrinsics, trajectory[frame_id], downsampling_step, depth_scale)
            all_points += points
            all_colors += colors
            print('Generated cloud for frame {}/{} (points {})'.format(frame_id + 1, len(rgb_list), len(all_points)))

        return all_points, all_colors

    @staticmethod
    def write_ply(filename, points, colors):
        print('Saving point cloud to {}'.format(filename))
        assert len(points) == len(colors)
        with open(filename, 'w') as file:
            file.write('ply\n')
            file.write('format ascii 1.0\n')
            file.write('element vertex {}\n'.format(len(points)))
            file.write('property float x\n')
            file.write('property float y\n')
            file.write('property float z\n')
            file.write('property uchar red\n')
            file.write('property uchar green\n')
            file.write('property uchar blue\n')
            file.write('end_header\n')

            for point, color in zip(points, colors):
                file.write('{} {}\n'.format(
                    ' '.join(map(str, point)),
                    ' '.join(map(str, color))
                ))

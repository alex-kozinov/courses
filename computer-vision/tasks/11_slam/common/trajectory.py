import numpy

from .dataset import Dataset


class Trajectory:
    _EPS = numpy.finfo(float).eps * 4.0
    FILE_HEADER = '#frame_id tx ty tz qx qy qz qw\n'

    @staticmethod
    def read(filename, key_type=int):
        trajectory_dict = Dataset.read_dict_of_lists(filename, key_type)

        trajectory = {}
        for key, pose_str in trajectory_dict.items():
            pose = [float(value) for value in pose_str]
            for value in pose:
                assert not numpy.isnan(value), 'Line "{}" of file "{}" has NaNs'.format(pose_str, filename)

            trajectory[key] = Trajectory.to_matrix4(pose)

        return trajectory

    @staticmethod
    def write(filename, trajectory):
        with open(filename, 'w') as file:
            file.write(Trajectory.FILE_HEADER)

            for frame_id, pose in trajectory.items():
                file.write('{} {}\n'.format(frame_id, ' '.join(map(str, Trajectory.from_matrix4(pose)))))

    @staticmethod
    def to_matrix4(positionAndQuaternion):
        """
        Generate matrix 4x4 homogeneous transformation matrix from matrix 3D point and unit quaternion.

        Input:
        positionAndQuaternion -- tuple consisting of (tx,ty,tz,qx,qy,qz,qw) where
             (tx,ty,tz) is the 3D position and (qx,qy,qz,qw) is the unit quaternion.

        Output:
        matrix -- 4x4 homogeneous transformation matrix
        """
        t = positionAndQuaternion[0:3]
        q = numpy.array(positionAndQuaternion[3:7], dtype=numpy.float64, copy=True)
        nq = numpy.dot(q, q)
        assert nq > Trajectory._EPS
        q *= numpy.sqrt(2.0 / nq)
        q = numpy.outer(q, q)
        return numpy.array((
            (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3], t[0]),
            (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3], t[1]),
            (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1], t[2]),
            (0.0, 0.0, 0.0, 1.0)
        ), dtype=numpy.float64)

    @staticmethod
    def from_matrix4(matrix):
        trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2] 
        if trace > 0:
            s = 0.5 / numpy.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (matrix[2, 1] - matrix[1, 2]) * s
            qy = (matrix[0, 2] - matrix[2, 0]) * s
            qz = (matrix[1, 0] - matrix[0, 1]) * s
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]: 
            s = 2.0 * numpy.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            qw = (matrix[2, 1] - matrix[1, 2]) / s
            qx = 0.25 * s
            qy = (matrix[0, 1] + matrix[1, 0]) / s
            qz = (matrix[0, 2] + matrix[2, 0]) / s
        elif matrix[1, 1] > matrix[2, 2]:
            s = 2.0 * numpy.sqrt( 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            qw = (matrix[0, 2] - matrix[2, 0]) / s
            qx = (matrix[0, 1] + matrix[1, 0]) / s
            qy = 0.25 * s
            qz = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = 2.0 * numpy.sqrt( 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1] )
            qw = (matrix[1, 0] - matrix[0, 1]) / s
            qx = (matrix[0, 2] + matrix[2, 0]) / s
            qy = (matrix[1, 2] + matrix[2, 1]) / s
            qz = 0.25 * s
            
        return [matrix[0, 3], matrix[1, 3], matrix[2, 3], qx, qy, qz, qw]

    @staticmethod
    def get_translations(trajectory):
        return numpy.matrix([matrix[0:3, 3] for matrix in trajectory.values()]).transpose()

    @staticmethod
    def align(model_trajectory, data_trajectory):
        """
        Align two trajectories using the method of Horn (closed-form).

        Output:
        rot -- rotation matrix (3x3)
        data_from_model_transform -- transform matrix (4x4)
        """

        model = Trajectory.get_translations(model_trajectory)
        data = Trajectory.get_translations(data_trajectory)

        numpy.set_printoptions(precision=3, suppress=True)
        model_zerocentered = model - model.mean(1)
        data_zerocentered = data - data.mean(1)

        W = numpy.zeros((3, 3))
        for column in range(model.shape[1]):
            W += numpy.outer(model_zerocentered[:, column], data_zerocentered[:, column])
        U, d, Vh = numpy.linalg.linalg.svd(W.transpose())
        S = numpy.matrix(numpy.identity(3))
        if numpy.linalg.det(U) * numpy.linalg.det(Vh) < 0:
            S[2, 2] = -1
        rot = U * S * Vh
        trans = data.mean(1) - rot * model.mean(1)

        data_from_model_transform = numpy.array((
            (rot[0, 0], rot[0, 1], rot[0, 2], trans[0]),
            (rot[1, 0], rot[1, 1], rot[1, 2], trans[1]),
            (rot[2, 0], rot[2, 1], rot[2, 2], trans[2]),
            (0.0, 0.0, 0.0, 1.0)
        ), dtype=numpy.float64)

        model_aligned = rot * model + trans
        alignment_error = model_aligned - data

        trans_error = numpy.sqrt(numpy.sum(numpy.multiply(alignment_error, alignment_error), 0)).A[0]

        return data_from_model_transform, trans_error

    @staticmethod
    def compute_relative_transform(matrix_to, matrix_from):
        return numpy.dot(numpy.linalg.inv(matrix_to), matrix_from)

    @staticmethod
    def compute_distance(matrix):
        return numpy.linalg.norm(matrix[0:3, 3])

    @staticmethod
    def compute_angle(matrix):
        return numpy.arccos(min(1, max(-1, (numpy.trace(matrix[0:3, 0:3]) - 1) / 2)))

    @staticmethod
    def transform_point(transform, point):
        return numpy.matmul(transform[0:3, 0:3], point) + transform[0:3, 3]

    @staticmethod
    def compute_median_distance_between_adjacent_frames(trajectory):
        keys = list(trajectory.keys())
        distances = []
        for id in range(0, len(trajectory) - 1):
            difference = Trajectory.compute_relative_transform(trajectory[keys[id]], trajectory[keys[id + 1]])
            distances.append(Trajectory.compute_distance(difference))

        return numpy.median(distances)

    @staticmethod
    def add_noise_to_translation(trajectory, sigma):
        result = trajectory.copy()
        for value in result.values():
            noise = numpy.random.multivariate_normal([0, 0, 0], numpy.diag([sigma ** 2] * 3))
            value[0:3, 3] += noise

        return result

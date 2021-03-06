import numpy

from .trajectory import Trajectory

class RelativePoseError:
    CHECK_WINDOW_WIDTH = 30  # about 1 second

    def __init__(self):
        self.translation_rmse = numpy.inf
        self.translation_mean = numpy.inf
        self.translation_median = numpy.inf
        self.translation_std = numpy.inf
        self.translation_min = numpy.inf
        self.translation_max = numpy.inf
        self.rotation_rmse = numpy.inf
        self.rotation_mean = numpy.inf
        self.rotation_median = numpy.inf
        self.rotation_std = numpy.inf
        self.rotation_min = numpy.inf
        self.rotation_max = numpy.inf

    @staticmethod
    def estimate(test_trajectory, gt_trajectory):
        assert len(test_trajectory) == len(gt_trajectory), 'Wrong trajectory length!'
        assert test_trajectory.keys() == gt_trajectory.keys()

        if len(test_trajectory) == 0:
            return RelativePoseError()

        keys = list(test_trajectory.keys())
        translation_errors = []
        rotation_errors = []
        for id in range(0, len(gt_trajectory) - RelativePoseError.CHECK_WINDOW_WIDTH):
            key = keys[id]
            ref_key = keys[id + RelativePoseError.CHECK_WINDOW_WIDTH]
            test_relative = Trajectory.compute_relative_transform(test_trajectory[key], test_trajectory[ref_key])
            gt_relative = Trajectory.compute_relative_transform(gt_trajectory[key], gt_trajectory[ref_key])

            difference = Trajectory.compute_relative_transform(gt_relative, test_relative)
            translation_errors.append(Trajectory.compute_distance(difference))
            rotation_errors.append(Trajectory.compute_angle(difference))

        result = RelativePoseError()
        result.translation_rmse = \
            numpy.sqrt(numpy.dot(translation_errors, translation_errors) / len(translation_errors))
        result.translation_mean = numpy.mean(translation_errors)
        result.translation_median = numpy.median(translation_errors)
        result.translation_std = numpy.std(translation_errors)
        result.translation_min = numpy.min(translation_errors)
        result.translation_max = numpy.max(translation_errors)

        result.rotation_rmse = \
            numpy.degrees(numpy.sqrt(numpy.dot(rotation_errors, rotation_errors) / len(rotation_errors)))
        result.rotation_mean = numpy.degrees(numpy.mean(rotation_errors))
        result.rotation_median = numpy.degrees(numpy.median(rotation_errors))
        result.rotation_std = numpy.degrees(numpy.std(rotation_errors))
        result.rotation_min = numpy.degrees(numpy.min(rotation_errors))
        result.rotation_max = numpy.degrees(numpy.max(rotation_errors))

        return result

import numpy

from .trajectory import Trajectory


class AbsoluteTranslationalError:
    def __init__(self):
        self.rmse = numpy.inf
        self.mean = numpy.inf
        self.median = numpy.inf
        self.std = numpy.inf
        self.min = numpy.inf
        self.max = numpy.inf

        self.gt_from_test_transform = None

    def _estimate_from_errors(self, errors):
        self.rmse = numpy.sqrt(numpy.dot(errors, errors) / len(errors))
        self.mean = numpy.mean(errors)
        self.median = numpy.median(errors)
        self.std = numpy.std(errors)
        self.min = numpy.min(errors)
        self.max = numpy.max(errors)

    @staticmethod
    def estimate(test_trajectory, gt_trajectory):
        assert len(test_trajectory) == len(gt_trajectory), 'Wrong trajectory length!'
        assert test_trajectory.keys() == gt_trajectory.keys()

        result = AbsoluteTranslationalError()
        if len(test_trajectory) == 0:
            return result

        result.gt_from_test_transform, errors = Trajectory.align(test_trajectory, gt_trajectory)
        result._estimate_from_errors(errors)

        return result

    def estimate_from_cloud(self, cloud, search_tree):
        errors = [
            search_tree.get_distance_to_nearest(Trajectory.transform_point(self.gt_from_test_transform, point))
            for point in cloud]
        result = AbsoluteTranslationalError()
        result._estimate_from_errors(errors)
        return result

#!/usr/bin/env python3

from json import dumps, load
import numpy
import os
from sys import argv, exit

from common.dataset import Dataset
from common import testing


def run_single_test(data_dir, output_dir):
    from estimate_trajectory import estimate_trajectory
    estimate_trajectory(data_dir, output_dir)


def check_test(data_dir):
    output_dir = os.path.join(data_dir, 'output')
    gt_dir = os.path.join(data_dir, 'gt')

    if os.environ.get('CHECKER'):
        result_file = '/tmp/scores.txt'
    else:
        result_file = Dataset.get_testing_result_file(output_dir)
    testing.test_result(output_dir, gt_dir, result_file)

    result = Dataset.read_dict_of_lists(result_file, key_type=str)
    inliers = float(result['inliers'])
    translation_error = float(result['absolute.translation.mean'])
    rotation_error = float(result['relative.rotation.mean'])

    res = f'Ok, {inliers:.2f} {translation_error:.2f} {rotation_error:.2f}'
    if os.environ.get('CHECKER'):
        print(res)
    return res


def grade(data_path):
    results = load(open(os.path.join(data_path, 'results.json')))
    if len(results) == 0:
        res = {'description': 'No datasets', 'mark': 0}
    else:
        all_ok = True
        worst_translation = 0.0
        worst_rotation = 0.0
        worst_inliers = 1.0
        for result in results:
            status = result['status']
            if not status.startswith('Ok'):
                all_ok = False
                res = {'description': '', 'mark': 0}
                break

            inliers, translation_error, rotation_error = \
                [float(v) for v in status[4:].split()]

            if inliers < testing.INLIERS_THRESHOLD:
                all_ok = False
                res = {'description': f'Too few inliers ({inliers})',
                       'mark': 0}
                break

            worst_translation = max(worst_translation, translation_error)
            worst_rotation = max(worst_rotation, rotation_error)
            worst_inliers = min(worst_inliers, inliers)

        if all_ok:
            translation_error = numpy.clip(
                (worst_translation - testing.GOOD_ENOUGH_DISTANCE) /
                (testing.DISTANCE_THRESHOLD - testing.GOOD_ENOUGH_DISTANCE),
                0, 1)
            rotation_error = numpy.clip(
                (numpy.radians(worst_rotation) - testing.GOOD_ENOUGH_ANGLE) /
                (testing.ANGLE_THRESHOLD - testing.GOOD_ENOUGH_ANGLE),
                0, 1)
            inliers_score = numpy.clip(
                (worst_inliers - testing.INLIERS_THRESHOLD) /
                (testing.GOOD_ENOUGH_INLIERS_COUNT - testing.INLIERS_THRESHOLD),
                0, 1)

            pose_score = 1.0 - 0.5 * (translation_error + rotation_error)
            inliers_exp_score = inliers_score ** 3

            mark = 20.0 * pose_score * inliers_exp_score
            res = {'description': f'tr {worst_translation:.2f} rot {worst_rotation:.2f}',
                   'mark': f'{mark:.2f}'}
    if os.environ.get('CHECKER'):
        print(dumps(res))
    return res


if __name__ == '__main__':
    if os.environ.get('CHECKER'):
        # Script is running in testing system
        if len(argv) != 4:
            print(f'Usage: {argv[0]} mode data_dir output_dir')
            exit(0)

        mode = argv[1]
        data_dir = argv[2]
        output_dir = argv[3]

        if mode == 'run_single_test':
            run_single_test(data_dir, output_dir)
        elif mode == 'check_test':
            check_test(data_dir)
        elif mode == 'grade':
            grade(data_dir)
    else:
        # Script is running locally, run on dir with tests
        if len(argv) != 2:
            print(f'Usage: {argv[0]} tests_dir')
            exit(0)

        from glob import glob
        from json import dump
        from re import sub
        from time import time
        from traceback import format_exc
        from os import makedirs
        from os.path import basename, exists, join
        from shutil import copytree

        tests_dir = argv[1]

        results = []
        for i, input_dir in enumerate(sorted(glob(join(tests_dir, '[0-9][0-9]_*_input')))):
            # if not i:
            #     continue
            output_dir = sub('input$', 'check', input_dir)
            run_output_dir = join(output_dir, 'output')
            makedirs(run_output_dir, exist_ok=True)
            gt_src = sub('input$', 'gt', input_dir)
            gt_dst = join(output_dir, 'gt')
            if not exists(gt_dst):
                copytree(gt_src, gt_dst)

            try:
                start = time()
                run_single_test(input_dir, run_output_dir)
                end = time()
                running_time = end - start
            except:
                status = 'Runtime error'
                traceback = format_exc()
            else:
                try:
                    status = check_test(output_dir)
                except:
                    status = 'Checker error'
                    traceback = format_exc()

            test_num = basename(input_dir)[:2]
            if status == 'Runtime error' or status == 'Checker error':
                print(test_num, status, '\n', traceback)
                results.append({'status': status})
            else:
                print(test_num, f'{running_time:.2f}s', status)
                results.append({
                    'time': running_time,
                    'status': status})

        dump(results, open(join(tests_dir, 'results.json'), 'w'))
        res = grade(tests_dir)
        print(f'Mark: {res["mark"]:.2f}', res['description'])

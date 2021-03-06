#!/usr/bin/env python3

from glob import glob
from json import load, dump, dumps
from os import environ
from os.path import basename, join, exists, splitext
from sys import argv
from skimage import img_as_ubyte
from skimage.io import imread, imsave

environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_iou(gt, pred):
    return (gt & pred).sum() / (gt | pred).sum()


def read_segm(fname):
    img = img_as_ubyte(imread(fname, as_gray=True))
    return img > 127


def check_test(data_dir):
    gt_dir = join(data_dir, 'gt')
    output_dir = join(data_dir, 'output')

    filenames = glob(join(gt_dir, '**/*.png'))

    res_iou = 0
    all_found = True
    for filename in filenames:
        name, ext = splitext(basename(filename))

        out_filename = join(output_dir, name + '.png')
        if not exists(out_filename):
            res = f'Error, segmentation for "{name}" not found'
            all_found = False
            res_iou = 0
            break

        pred_segm = read_segm(out_filename)
        gt_segm = read_segm(filename)
        iou = get_iou(gt_segm, pred_segm)
        res_iou += iou

    res_iou /= len(filenames)
    if all_found:
        res = f'Ok, IoU {res_iou:.4f}'

    if environ.get('CHECKER'):
        print(res)
    return res


def grade(data_path):
    results = load(open(join(data_path, 'results.json')))
    result = results[-1]['status']

    if not result.startswith('Ok'):
        res = {'description': '', 'mark': 0}
    else:
        iou_str = result[8:]
        iou = float(iou_str)

        if iou >= 0.85:
            mark = 10
        elif iou >= 0.80:
            mark = 9
        elif iou >= 0.75:
            mark = 8
        elif iou >= 0.70:
            mark = 7
        elif iou >= 0.60:
            mark = 6
        elif iou >= 0.50:
            mark = 5
        elif iou >= 0.40:
            mark = 4
        elif iou >= 0.30:
            mark = 3
        elif iou >= 0.2:
            mark = 2
        else:
            mark = 0

        res = {'description': iou_str, 'mark': mark}
    if environ.get('CHECKER'):
        print(dumps(res))
    return res


def run_single_test(data_dir, output_dir):
    from segmentation import predict
    from tensorflow.keras.models import load_model
    from os.path import abspath, basename, dirname, join

    code_dir = dirname(abspath(__file__))

    # from segmentation import train_segmentation_model
    # model = train_segmentation_model(join(data_dir, 'train'))

    model = load_model(join(code_dir, 'segmentation_model.hdf5'), compile=False)

    img_filenames = glob(join(data_dir, 'test/images/**/*.jpg'))

    for filename in img_filenames:
        segm = (predict(model, filename) > 0.5).astype('uint8') * 255
        name, ext = splitext(basename(filename))
        imsave(join(output_dir, name + '.png'), segm)


if __name__ == '__main__':
    if environ.get('CHECKER'):
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

        from re import sub
        from time import time
        from traceback import format_exc
        from os import makedirs
        from shutil import copytree

        tests_dir = argv[1]

        results = []
        for input_dir in sorted(glob(join(tests_dir, '[0-9][0-9]_*_input'))):
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
        print('Mark:', res['mark'], res['description'])

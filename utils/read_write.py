import os
import csv
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch

from utils import logger, const, getter


def read_csv(path, index=None):
    data = []

    try:
        with open(path, 'r') as f:
            reader = csv.reader(f)
            _ = next(reader)
            for row in reader:
                if index is None:
                    data.append(row)
                else:
                    data.append(float(row[index]))
        return data
    except Exception as e:
        raise e


def write_csv(data, folder, file, mode='w'):
    Path(folder).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(folder, file), mode, newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(data)
    f.close()


def write_npy(data, folder, file):
    Path(folder).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(folder, file), data)


def write_pt(data, folder, file):
    Path(folder).mkdir(parents=True, exist_ok=True)
    torch.save(data, os.path.join(folder, file))


def read_acc_files(args):
    l = logger.Logger.get()
    acc_dict = OrderedDict()
    acc_mean_dict = OrderedDict()
    acc_std_dict = OrderedDict()
    files = getter.get_result_files(args)
    methods = list(files.keys())
    run_counts = {}
    
    for method in methods:
        run_counts[method] = [i for i in range(len(files[method]))]
        acc_dict[method] = OrderedDict()

    for method in methods:
        paths = files[method]

        for run in run_counts[method]:
            file_path = paths[run]
            l.info(f'loading file: {file_path}\n')

            acc_dict[method][run] = np.array(read_csv(file_path, 1)) * 100

        concat_array = np.vstack([acc_dict[method][run]]
                                 for run in run_counts[method])

        acc_mean_dict[method] = np.mean(concat_array, axis=0)
        acc_std_dict[method] = np.std(concat_array, axis=0)

    return acc_mean_dict, acc_std_dict

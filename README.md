# Hybrid Active Learning with Uncertainty-Weighted Embeddings

This is the PyTorch implementation of "Hybrid Active Learning with Uncertainty-Weighted Embeddings".

## Setup

Use [pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel](https://hub.docker.com/layers/pytorch/pytorch/1.7.1-cuda11.0-cudnn8-devel/images/sha256:f0d0c1b5d4e170b4d2548d64026755421f8c0df185af2c4679085a7edc34d150) docker image and run `pip install scipy scikit-learn cython`.

To use CUDA to compute Euclidean distance, navigate to `utils/cuda` directory and run `python setup.py build_ext --inplace`.

CIFAR-10 and CIFAR-100 are automatically downloaded. For mini-ImageNet, manually download `mini-imagenet-cache-train.pkl`, `mini-imagenet-cache-val.pkl`, and `mini-imagenet-cache-test.pkl` files and save them under `dataset/miniimagenet` directory path.

## Run Experiments

To run active learning experiments, run `main.py`.

- `--action`
  - `initialize` to generate cycle 0 labeled indexes and validation indexes (if any).
  - `alc0` to train the cycle 0 initial model.
  - `al` to run active learning cycles.
- `--method`: Active learning methods, `uwe` for our UWE.
- `--dataset`: `cifar10`, `cifar100`, or `mini_imagenet`.
- `--seed`: Seed, default set to 1.
- `--budget_id`: Active learning cycle budgets are stored in `utils/const.py`.
- `--model`: Model architecture.
- `--n_epoch`: Number of epochs for model training.

Available Methods:

1. UWE (`uwe`)
2. Random sampling (`random`)
3. Entropy sampling (`entropy`)
4. Best vs second best (`bvsb`)
5. Least confident (`least_confident`)
6. Core-Set (`coreset`)
7. BADGE (`badge`)
8. BALD (`bald`)

### Using CIFAR-10 example

1. Initialize Cycle 0 Indexes

    ```sh
    python main.py --action initialize --dataset cifar10 --seed 1 --budget_id exp05k --n_epoch 100
    ```

2. Train Cycle 0

    ```sh
    python main.py --action alc0 --dataset cifar10 --seed 1 --budget_id exp05k --model resnet18 --gpu_id 0 --n_epoch 100
    ```

3. Active Learning

    ```sh
    python main.py --action al --method uwe --dataset cifar10 --seed 1 --budget_id exp05k --model resnet18 --gpu_id 0 --n_epoch 100
    ```

## Experimental results

All results, including logs, extracted features and model uncertainties, are saved in `results/{dataset}/{al method}` directory.

Active learning cycles results are saved in `results/{dataset/{al method}/al_cycle_result/` directory with file naming convention of `{dataset}_al_mod{model}_s{seed}_ne{number of epochs}_bid{budget id}_m{al method}_al_cycle_result.csv`.

To calculate mean and standard deviation of classification accuracy across different runs, record active learning result file paths in `utils/const.py` in the dictionary format shown below and use the `read_acc_files` function in `utils/read_write.py`.

```py
al_result_files = {
    '{dataset name}': {
        '{budget id}': {
            '{al method}': [
                # .csv files storing active learning cycle results
            ],
        }
    },
}
```

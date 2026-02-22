# Deep Translation Averaging

This repository provideds the code associated wth our paper:
DeepTA: High-Speed Deep Camera Translation Averaging with Reverse Direction Invariance

## Dependencies

- torch-cluster==1.6.0
- torch-geometric==2.2.0
- torch-scatter==2.1.0
- torch-sparse==0.6.16

`pip install -r requirements.txt`

## Usage

### Data Preparation

The data should be stored in .mat format. 
Each graph should contain the following keys:

- `inds (Ex2)`: the indices of the relative translation
- `rel_t (Ex3)`: the corresponding relative translation direction
- `t_gt (Nx3)`: the ground truth camera locations

### Experimental Eetup

All experimental settings are specified in the yaml config file.

Please refer to ./configs/default.yml for an example.

### Implementation

For training: 

```sh
python train.py --configs config.yaml
```

For testing:
```sh
python test.py --configs config.yaml
```


## Citation

```
@article{liu2026deeptA,
  title={DeepTA: High-Speed Deep Camera Translation Averaging with Reverse Direction Invariance},
  author={Liu, Y. and Dong, Q.},
  journal={International Journal of Computer Vision},
  volume={134},
  pages={132},
  year={2026},
  doi={10.1007/s11263-025-02714-x}
}
```

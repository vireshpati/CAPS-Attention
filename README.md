# CAPS: Unifying Attention, Recurrence, and Alignment in Transformer-based Time Series Forecasting

Official implementation of the paper: https://arxiv.org/pdf/2602.02729v1

## Installation

Install [PyTorch](https://pytorch.org/get-started/locally/) first, then:

```bash
pip install -r requirements.txt
```

## Datasets

All datasets are expected in the `dataset/` directory. The 10 datasets used in the paper are:

| Dataset | Channels |
|---------|----------|
| ETTm1   | 7        |
| ETTm2   | 7        |
| ETTh1   | 7        |
| ETTh2   | 7        |
| Weather | 21       |
| Solar   | 137      |
| ECL     | 321      |
| PEMS03  | 358      |
| PEMS04  | 307      |
| PEMS08  | 170      |

These can be sourced from [Time-Series-Library](https://github.com/thuml/Time-Series-Library).

## Running Experiments

**Full CAPS model:**

```bash
bash caps.sh
```

**LinAttn + RoPE ablation baseline:**

```bash
bash caps_baseline.sh
```


## Monitoring

Metrics are logged to [Weights & Biases](https://wandb.ai). Set `WANDB_MODE=disabled` to run offline.

# Citation

```
@misc{pati2026capsunifyingattentionrecurrence,
      title={CAPS: Unifying Attention, Recurrence, and Alignment in Transformer-based Time Series Forecasting}, 
      author={Viresh Pati and Yubin Kim and Vinh Pham and Jevon Twitty and Shihao Yang and Jiecheng Lu},
      year={2026},
      eprint={2602.02729},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.02729}, 
}
```

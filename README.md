# Session-based Hotel Recommandation based on the Trivago Dataset


Contact: marton.szep@tum.de

## Introduction
This repository is based on RosettaAI's approach to the 2019 ACM Recys Challenge ([paper](https://dl.acm.org/citation.cfm?id=3359560), [writeup](https://medium.com/@huangkh19951228/the-5th-place-approach-to-the-2019-acm-recsys-challenge-by-team-rosettaai-eb3c4e6178c4)). Instead of treating the task as a ranking problem, it is formulated as a Binary Cassification task. Four different models were implemented: two of them use [GRU](https://arxiv.org/pdf/1406.1078.pdf) modules and the other two are based on [Transformer](https://arxiv.org/abs/1706.03762) blocks. For more information see the `src/nn.py` file.


## Environment
Requirements can be found in `requirements.txt` and can be installed in a conda environment using:
```bash
conda create -n <myenv> python=3.7
conda activate <myenv>
pip install -r requirements.txt
```

## Project Structure

```
├── input
├── output
├── src
└── weights
```

## Setup
Run the following commands to create directories that conform to the structure of the project, then place the unzipped data into the ```input``` directory:

```bash
. setup.sh
```

Run the two python scripts to picklize the input data and obtain the utc offsets from countries:
```bash
cd src
python picklization.py
python country2utc.py
```

To enable the model to train on the whole data, set ```debug``` to ```False``` and ```subsample``` to ```1``` in the ```src/config.py``` file.

```python
class Configuration(object):

    def __init__(self):
        ...
        self.debug = False
        self.sub_sample = 1.
        ...
```


## Training & Submission

The models are all trained in an end-to-end fashion. To train a model and evaluate it ont the test set, simply run the following commands:
```bash
python run_nn.py
```
The submission files are stored in the ```output``` directory. The first time you run a training, the preprocessed data will be saved in the ```input``` directory as a pickle file. Later, you don't need to run the data preprocessing pipeline anymore, you can just set ```load_preproc_data``` to ````True``` in the ```src/config.py``` file.

## Performance

| Model        | Validation MRR           | Dataset Proportion Used  |
| ------------- |-------------:| -----:|
| [Layer 6 AI](https://dl.acm.org/doi/10.1145/3359555.3359558)*      | 0.676500 | 1 |
| [RosettaAI](https://dl.acm.org/citation.cfm?id=3359560)*      | 0.675206 | 1 |
| GruNet1      | 0.667437 | 0.25 |
| GruNet2    | 0.668033 |   0.25  |
| TransformerNet1 | 0.659480      |    0.25  |
| TransformerNet2 | 0.660470     |    0.25  |

*Best performing Deep Learning model 

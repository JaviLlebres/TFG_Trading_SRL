# Manual features vs Learned representations in algorithmic trading

## Prerequisites
- Python 3.12

## Create `venv`
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
## Install dependencies
```bash
pip install -r requirements.txt
```

## Run the notebooks

Ru the notebooks in the `notebooks` folder in the number and letters order. The notebooks are designed to be run in sequence, and each notebook builds on the previous one in most of the cases. The final notebook is the main one that contains the results of the analysis.

## Data

For this repo, BTCUSDT, ETHUSDT and SP&500 are the only dataset used, but others can be added, make sure to add them in `data` and in a folder and rename it as the others.

## Results

Due to the size of the results, they are not included in the repo. However, you can reproduce the results by running the notebooks in the `notebooks` folder. The results will be saved in the `RL_outputs/results/` folder.



# Robust Time Series Analysis with Topological Data Analysis and Deep Reinforcement Learning

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

For this repo, ADAUSDT is the only dataset used. The data is downloaded from Binance using the `binance` package. The data is stored in the `data` folder.

## Results

Due to the size of the results, they are not included in the repo. However, you can reproduce the results by running the notebooks in the `notebooks` folder. The results will be saved in the `RL_outputs/results/` folder.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Docs

- Supplementary materials for the paper: [SupMat](docs/SupMat.pdf).
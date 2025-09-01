# precipitation-forecasting

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Implementation of a predictive model based of LSTM networks for precipitation timeseries forecasting in Cuba.

**Key features**
- Architecture: LSTM layer (24 neurons) + LSTM layer (36 neurons) + Dense (12 neurons)
- Preprocessing: Temporal interpolation + scaling [-1,1]
- Metrics:
      MAE: 0.26
      RMSE: 0.32
      Bias: 0.05

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks
│
├── reports         
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment.
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes eda a Python module
    │
    ├── eda.py               <- Code to explorate data analysis module
    │
    ├── missing_data.py              <- Code for missing data analysis
    │
    ├── preprocess.py             <- Code for data loading and preprocessing for forecasting reate features for modeling
    │
    ├── ts_trainer.py             <- Utilities for timesereis forecasting
    │
    └── visualization.py                <- Code to create visualizations
```

--------
**Would you like any modification to this version?**

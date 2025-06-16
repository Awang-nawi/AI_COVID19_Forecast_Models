# AI\_COVID19\_Forecast\_Models

Forecasting Malaysia’s daily COVID-19 cases from 1 April 2021 to 18 September 2021 using ARIMA, LSTM, and a hybrid ARIMA-LSTM model.

## Dataset Description

* **File**: `RESEARCH DATA.csv`

* **Format**: CSV file with the following columns:

  * `date`: Date in `YYYY-MM-DD` format
  * `cases`: Daily confirmed COVID-19 cases in Malaysia

* **Preprocessing**: Date is parsed automatically by scripts; no manual preprocessing required.

## Environment Setup

To run the models, follow these steps:

1. **Clone the repository**:

```bash
git clone https://github.com/Awang-nawi/AI_COVID19_Forecast_Models.git
cd AI_COVID19_Forecast_Models
```

2. **Install required dependencies**:

```bash
pip install -r requirements.txt
```

## Running the Models

### ARIMA Model

```bash
python arima_model.py --input data/data.csv --output results/arima_forecast.csv
```

### LSTM Model

```bash
python lstm_model.py --input data/data.csv --output results/lstm_forecast.csv
```

### Hybrid ARIMA-LSTM Model

```bash
python hybrid_model.py --input data/data.csv --output results/hybrid_forecast.csv
```

## Output Files

* Forecast results saved as `.csv` files in the `results/` directory.
* Plots (time series, error metrics) saved in `results/plots/`.

## Requirements

Dependencies are listed in `requirements.txt`. Key packages include:

* `pandas`
* `numpy`
* `statsmodels`
* `matplotlib`
* `scikit-learn`
* `torch` (for LSTM model)

## Reproducibility

* Recommended Python version: **3.8+**
* To ensure consistent results in LSTM-based models, you can use a random seed:

```bash
python lstm_model.py --seed 42
```

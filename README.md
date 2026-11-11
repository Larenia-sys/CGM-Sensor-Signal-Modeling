# CGM Electrochemical Sensor Signal Modeling & Hypoglycemia Prediction

This project provides an end-to-end data pipeline for Continuous Glucose Monitoring (CGM) data. It processes raw data from FreeStyle Libre 2 electrochemical enzymatic sensors (glucose oxidase) and builds predictive models for glucose forecasting and hypoglycemia early warning.

## Dataset
- HUPA-UCM Diabetes Dataset (Hidalgo et al., 2024), Mendeley Data. (CC BY 4.0)
- 25 Type 1 Diabetes patients, 15-min sampling interval.

## Core Features
1. **Robust Data Parsing (`io.py`)**: Handles 3 different generations of LibreView export formats (various separators, encodings, CRLF). Fixes time-axis breaks caused by sensor disconnections or device resets.
2. **Signal Quality Assessment (`quality.py`)**: Detects "compression lows" (false hypoglycemia caused by physical pressure on the sensor during sleep).
3. **Feature Engineering (`features.py`)**: Extracts time-series features (lags, rolling statistics, circadian rhythms). Strictly uses `groupby(segment)` to prevent data leakage across time gaps.
4. **Patient-Level Validation (`model.py`)**: Uses GroupKFold (grouped by patient) instead of standard KFold to prevent individual baseline memorization.
5. **Clinical Evaluation (`metrics.py`)**: Moves beyond RMSE to use Clarke Error Grid and event-level hypoglycemia recall.
6. **Hypoglycemia Prediction (`hypo.py`)**: Reframes the regression problem as an imbalanced classification task to address the regression-to-the-mean issue in low glucose ranges.

## How to Run
1. Download the dataset and put the `.csv` files into `data/raw/`.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the pipeline: `python scripts/run_pipeline.py --raw-dir data/raw --out reports`

## Results
*(You can drag and drop your generated figures here later)*

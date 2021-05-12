# ds_practical
Data Science Practical project from Siemens on Detecting Concept Drift in Small Time Series Data.

## Task
Explicit detection of concept drifts in small time series data, with subsequent incorporation into forecasting routines.

## Artificial Time Series
Random time series according to those models described in the Report can be generated through the functions in [create_simdata.py](create_simdata.py). Additionally to sudden drifts, there is also the option to generate data with incremental drifts. Some functions to generate these time series are implemented in [functions.py](functions.py).

## Drift detection
Drift detection functions used for the report are implemented in different ways in [functions.py](functions.py). Various functions are extracting features used for the RBF kernel (see e.g. mutual_info() or autocorrelations_in_window()). The function preprocess_timeseries() prepares a time series to be used for the RBF kernel. The final drift detection analysis is implemented in our analysis_X methods. There, mean (l2), linear (ar) and RBF drift detection is applied to multiple different time series and statistics are collected over all iterations. The functions analysis_X_final use time series  previously generated through the function create_final_data() (analysis_X were earlier random versions). Finally stability_analysis_long_term() and stability_analysis_short_term() are experimental functions to check the stability of our break point location, but need more work.

Our hyperparameter optimization and the analysis used in the report are then implemented in  [drift_detection.py](drift_detection.py) using the previously described functions.

## Forecasting
All the necessary dependencies for the forecasting can be installed from [environment.yml](global_run/environment.yml).

The data for forecasting is already stored in the **data** folder, but can be overwritten with new one by running [create_same_data.py](create_same_data.py).

Jupyter notebooks represent an examplary code mostly designed for testing and debugging.

The folder [global_run](global_run) contains the scripts for producing several sets of results for all time series.  

# ds_practical
Data Science Practical project from Siemens on Detecting Concept Drift in Small Time Series Data

## Task
Explicit detection of concept drifts in small time series data, with subsequent incorporation into forecasting routines.

## Artificial Time Series
Random time series according to those models described in the Report can be generated through the functions in [create_simdata.py](create_simdata.py). Additionally to sudden drifts, there is also the option to generate data with incremental drifts.

## Drift detection

## Forecasting
All the necessary dependencies for the forecasting can be installed from [environment.yml](global_run/environment.yml).

The data for forecasting is already stored in the **data** folder, but can be overwritten with new one by running [create_same_data.py](create_same_data.py).

Jupyter notebooks represent an examplary code mostly designed for testing and debugging.

The folder [global_run](global_run) contains the scripts for producing several sets of results for all time series.  

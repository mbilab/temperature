# Temperature Prediction with Neural Network and Gradient Boosting

## 1. Objective

This project aims to predict future hourly temperatures using past weather data.

## 2. Data Collection

Raw weather data is extracted from [CWB Observation Data Inquire System](https://e-service.cwb.gov.tw/HistoryDataQuery/index.jsp). The crawler locates at `nn/data/getdata.py`, where `crawl(startDate, endDate, path)` is used to get `.csv` files of hourly weather data for the given time interval. For instance:

```python
crawlCSV((2010, 1, 1), (2018, 12, 31), 'raw.csv')
```

## 3. Data Processing

To get numpy arrays, one can call `getXY(path, inputHrs, hrsAfter, features)` in `nn/data/getdata.py`, where

 * `path` specifies the path to a `.csv` file
 * `inputHrs` specifies how many hours are used as input features
 * `hrsAfter` specifies how many hours later to predict after the last input hour
 * `features` specifies the features used for each hour

For instance,

```python
x, y = getXY('data/raw.csv', inputHrs=72, hrsAfter=24, features=['Temperature','StnPres'])
```

returns numpy arrays `x, y` with shape `(N, 72, 2), (N, )`, respectively.  `N` is the amount of data.

For XGBoost, data preprocessing is in `xgboost/get_data.py`. Use

```python
from_2010(path, tag, train_hour, max_train_hour, pred_hour)
```

to get raw features e.g. temperature (`tag` is the column index). Use

```python
get_time(path, max_hour, pred_hour)
```

to get normalized date and time, where date is divided by 366, time is devided by 24.

## 4. Model Training

In `./nn/`, each `.py` file represents a model. For instance, one can call `CNNmodel(xtrain, ytrain)` in `./nn/CNNmodel.py` to train the model. Models may have different shapes of `xtrain`. See `./nn/*.py` for details. Note that, after training is done, only loss history is reported. The best model is saved to `./nn/models/` with **ModelCheckpoint** and **EarlyStopping**. Use `m = load_model(path)` to get the trained model. Below are models that have been tested.

* **CNNmodel:** Basic CNN without pooling. Only **Temperature** is used as the input feature.
* **CNNmodel_v2:** CNN with 3 input feature (stored in 3 channels): **Temperature**, **Td dew point** and **StnPres**.
* **CNNconcatFeature:** The same as **CNNmodel_v2**, but the 3 features are concatenated.
* **CNNsepFeature:** The same as  **CNNmodel_v2**, but the 3 features are submitted to 3 individual CNNs (see Figure (a)).
* **CNNwithMonth:** Only **Temperature** is submitted to a CNN. Circularly encoded month is concatenated after the CNN (see in figure (b)).
* **LSTMmodel:** LSTM with **Temperature** as the input feature.
* **TrivialModel:** This model just outputs the last input **Temperature**, just for comparison with other models.

|**(a) CNNsepFeature**| **(b) CNNwithMonth**    |
|---------------------|-------------------------|
|![](./img/sepCNN.PNG)|![](./img/CNNwithMon.PNG)|

For XGBoost (see `./xgboost/`):

* **train.py**: The features may be raw or normalized.
* **train_cross.py**: Cross validation and testing for each year.
* **train_PCA.py**: The features underwent PCA.
* **tuning.py**: Tune parameters for raw/normalized features.
* **tuning_PCA.py**: Tune parameters for PCA features.

## 5. Results

We used mean-absolute-error (MAE) in degrees Celsius as the metric to evaluate models. Below are the performance:

| Model              | MAE |
|--------------------|-----|
|**TrivialModel**    |1.920|
|**CNNmodel**        |1.398|
|**CNNmodel_v2**     |1.402|
|**CNNwithMonth**    |1.350|
|**CNNconcatFeature**|1.399|
|**CNNsepFeature**   |1.444|
|**LSTM**            |1.414|

Some additional weather features, such as pressure and humidity (**CNNsep**), but temperature is still the most dominent feature. Adding month as a feature slightly improved the model (**CNNwithMonth**). Our models performed particularly bad between winter and spring as shwon in the plot below. The error in the first few months are relatively high.

![](/img/bigerror.PNG)

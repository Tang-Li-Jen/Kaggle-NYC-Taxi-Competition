

```python
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn import linear_model,metrics,svm
from sklearn.ensemble import RandomForestRegressor
from geopy.distance import vincenty,great_circle
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import requests
import time
import holidays
```


```python
#import training and testing dataframe
train = pd.read_csv('../charlie/Desktop/NYC_Taxi/train.csv')
test = pd.read_csv('../charlie/Desktop/NYC_Taxi/test.csv')

#check the numbers of dataframe
if len(train) == 1458644:
    print 'numbers of training data :correct!'
else:
    print "numbers of training data :incorrect!"

if len(test) == 625134:
    print 'numbers of testing data :correct!'
else:
    print "numbers of testing data :incorrect!"
    
# for convinience,make them together to do futher process
df = pd.concat([train,test])
print len(df)
df.head()
```

    numbers of training data :correct!
    numbers of testing data :correct!



```python
# desccriptive statistics
print len(train.columns.values)
print len(test.columns.values)
print len(df.columns.values)
```


```python
#time
df['p_date'] = pd.to_datetime(df['pickup_datetime'])
df['d_date'] = pd.to_datetime(df['dropoff_datetime'])
df['weekday'] = df.p_date.dt.weekday
df.loc[df.weekday >= 5, 'is_weekend'] = 1
df.loc[df.weekday < 5, 'is_weekend'] = 0
df['p_hour'] = df.p_date.dt.hour
df.loc[(df.p_hour >=0 ) & (df.p_hour <6),'time_period'] = 1
df.loc[(df.p_hour >=6 ) & (df.p_hour <12),'time_period'] = 2
df.loc[(df.p_hour >=12 ) & (df.p_hour <18),'time_period'] = 3
df.loc[(df.p_hour >=18 ) & (df.p_hour <24),'time_period'] = 4
df['month'] = df['p_date'].dt.month
df['weekofyear'] = df['p_date'].dt.weekofyear

#distance
def D(x,y,z,w):
    return vincenty((x, y),(z, w)).miles
df['distance'] = map(D,df.pickup_latitude,df.pickup_longitude,df.dropoff_latitude,df.dropoff_longitude)

#trip_duration
df['trip_duration2'] = np.log(df.trip_duration.values)
df.head(5)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dropoff_datetime</th>
      <th>dropoff_latitude</th>
      <th>dropoff_longitude</th>
      <th>id</th>
      <th>passenger_count</th>
      <th>pickup_datetime</th>
      <th>pickup_latitude</th>
      <th>pickup_longitude</th>
      <th>store_and_fwd_flag</th>
      <th>trip_duration</th>
      <th>vendor_id</th>
      <th>p_date</th>
      <th>d_date</th>
      <th>weekday</th>
      <th>is_weekend</th>
      <th>p_hour</th>
      <th>time_period</th>
      <th>month</th>
      <th>weekofyear</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-03-14 17:32:30</td>
      <td>40.765602</td>
      <td>-73.964630</td>
      <td>id2875421</td>
      <td>1</td>
      <td>2016-03-14 17:24:55</td>
      <td>40.767937</td>
      <td>-73.982155</td>
      <td>N</td>
      <td>455.0</td>
      <td>2</td>
      <td>2016-03-14 17:24:55</td>
      <td>2016-03-14 17:32:30</td>
      <td>0</td>
      <td>0.0</td>
      <td>17</td>
      <td>3.0</td>
      <td>3</td>
      <td>11</td>
      <td>0.933406</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-06-12 00:54:38</td>
      <td>40.731152</td>
      <td>-73.999481</td>
      <td>id2377394</td>
      <td>1</td>
      <td>2016-06-12 00:43:35</td>
      <td>40.738564</td>
      <td>-73.980415</td>
      <td>N</td>
      <td>663.0</td>
      <td>1</td>
      <td>2016-06-12 00:43:35</td>
      <td>2016-06-12 00:54:38</td>
      <td>6</td>
      <td>1.0</td>
      <td>0</td>
      <td>1.0</td>
      <td>6</td>
      <td>23</td>
      <td>1.123849</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-01-19 12:10:48</td>
      <td>40.710087</td>
      <td>-74.005333</td>
      <td>id3858529</td>
      <td>1</td>
      <td>2016-01-19 11:35:24</td>
      <td>40.763939</td>
      <td>-73.979027</td>
      <td>N</td>
      <td>2124.0</td>
      <td>2</td>
      <td>2016-01-19 11:35:24</td>
      <td>2016-01-19 12:10:48</td>
      <td>1</td>
      <td>0.0</td>
      <td>11</td>
      <td>2.0</td>
      <td>1</td>
      <td>3</td>
      <td>3.964154</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-04-06 19:39:40</td>
      <td>40.706718</td>
      <td>-74.012268</td>
      <td>id3504673</td>
      <td>1</td>
      <td>2016-04-06 19:32:31</td>
      <td>40.719971</td>
      <td>-74.010040</td>
      <td>N</td>
      <td>429.0</td>
      <td>2</td>
      <td>2016-04-06 19:32:31</td>
      <td>2016-04-06 19:39:40</td>
      <td>2</td>
      <td>0.0</td>
      <td>19</td>
      <td>4.0</td>
      <td>4</td>
      <td>14</td>
      <td>0.921886</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-03-26 13:38:10</td>
      <td>40.782520</td>
      <td>-73.972923</td>
      <td>id2181028</td>
      <td>1</td>
      <td>2016-03-26 13:30:55</td>
      <td>40.793209</td>
      <td>-73.973053</td>
      <td>N</td>
      <td>435.0</td>
      <td>2</td>
      <td>2016-03-26 13:30:55</td>
      <td>2016-03-26 13:38:10</td>
      <td>5</td>
      <td>1.0</td>
      <td>13</td>
      <td>3.0</td>
      <td>3</td>
      <td>12</td>
      <td>0.737591</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2016-01-30 22:09:03</td>
      <td>40.749184</td>
      <td>-73.992081</td>
      <td>id0801584</td>
      <td>6</td>
      <td>2016-01-30 22:01:40</td>
      <td>40.742195</td>
      <td>-73.982857</td>
      <td>N</td>
      <td>443.0</td>
      <td>2</td>
      <td>2016-01-30 22:01:40</td>
      <td>2016-01-30 22:09:03</td>
      <td>5</td>
      <td>1.0</td>
      <td>22</td>
      <td>4.0</td>
      <td>1</td>
      <td>4</td>
      <td>0.683275</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2016-06-17 22:40:40</td>
      <td>40.765896</td>
      <td>-73.957405</td>
      <td>id1813257</td>
      <td>4</td>
      <td>2016-06-17 22:34:59</td>
      <td>40.757839</td>
      <td>-73.969017</td>
      <td>N</td>
      <td>341.0</td>
      <td>1</td>
      <td>2016-06-17 22:34:59</td>
      <td>2016-06-17 22:40:40</td>
      <td>4</td>
      <td>0.0</td>
      <td>22</td>
      <td>4.0</td>
      <td>6</td>
      <td>24</td>
      <td>0.824764</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2016-05-21 08:20:49</td>
      <td>40.760559</td>
      <td>-73.922470</td>
      <td>id1324603</td>
      <td>1</td>
      <td>2016-05-21 07:54:58</td>
      <td>40.797779</td>
      <td>-73.969276</td>
      <td>N</td>
      <td>1551.0</td>
      <td>2</td>
      <td>2016-05-21 07:54:58</td>
      <td>2016-05-21 08:20:49</td>
      <td>5</td>
      <td>1.0</td>
      <td>7</td>
      <td>2.0</td>
      <td>5</td>
      <td>20</td>
      <td>3.553009</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2016-05-27 23:16:38</td>
      <td>40.732815</td>
      <td>-73.985786</td>
      <td>id1301050</td>
      <td>1</td>
      <td>2016-05-27 23:12:23</td>
      <td>40.738400</td>
      <td>-73.999481</td>
      <td>N</td>
      <td>255.0</td>
      <td>1</td>
      <td>2016-05-27 23:12:23</td>
      <td>2016-05-27 23:16:38</td>
      <td>4</td>
      <td>0.0</td>
      <td>23</td>
      <td>4.0</td>
      <td>5</td>
      <td>21</td>
      <td>0.815587</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2016-03-10 22:05:26</td>
      <td>40.789989</td>
      <td>-73.973000</td>
      <td>id0012891</td>
      <td>1</td>
      <td>2016-03-10 21:45:01</td>
      <td>40.744339</td>
      <td>-73.981049</td>
      <td>N</td>
      <td>1225.0</td>
      <td>2</td>
      <td>2016-03-10 21:45:01</td>
      <td>2016-03-10 22:05:26</td>
      <td>3</td>
      <td>0.0</td>
      <td>21</td>
      <td>4.0</td>
      <td>3</td>
      <td>10</td>
      <td>3.178194</td>
    </tr>
  </tbody>
</table>
</div>




```python
# use Open Street Road Map(OSRM)
#import data
fr_train1 = pd.read_csv('../charlie/Desktop/NYC_Taxi/fastest_routes_train_part_1.csv')
fr_train2 = pd.read_csv('../charlie/Desktop/NYC_Taxi/fastest_routes_train_part_2.csv')
fr_test = pd.read_csv('../charlie/Desktop/NYC_Taxi/fastest_routes_test.csv')
fr_train = pd.concat([fr_train1,fr_train2])
print len(fr_train)
fr = pd.concat([fr_train,fr_test])
print len(fr)
df = df.merge(fr, how='left', on='id')
df['speed'] = df['total_distance']/df['total_travel_time']
print len(df)
```

    1458643
    2083777
    2083778



```python
# coordinate:dimention reduction
coords = np.vstack((df[['pickup_latitude', 'pickup_longitude']].values,
                    df[['dropoff_latitude', 'dropoff_longitude']].values))
pca = PCA().fit(coords)
df['pickup_pca0'] = pca.transform(df[['pickup_latitude', 'pickup_longitude']])[:,0]
df['pickup_pca1'] = pca.transform(df[['pickup_latitude', 'pickup_longitude']])[:,1]
df['dropoff_pca0'] = pca.transform(df[['dropoff_latitude', 'dropoff_longitude']])[:,0]
df['dropoff_pca1'] = pca.transform(df[['dropoff_latitude', 'dropoff_longitude']])[:,1]
#clustering
#sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords)
df['pickup_group'] = kmeans.predict(df[['pickup_latitude', 'pickup_longitude']])
df['dropoff_group'] = kmeans.predict(df[['dropoff_latitude', 'dropoff_longitude']])
```


```python
#split data into training and testing data

```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dropoff_datetime</th>
      <th>dropoff_latitude</th>
      <th>dropoff_longitude</th>
      <th>id</th>
      <th>passenger_count</th>
      <th>pickup_datetime</th>
      <th>pickup_latitude</th>
      <th>pickup_longitude</th>
      <th>store_and_fwd_flag</th>
      <th>trip_duration</th>
      <th>...</th>
      <th>step_maneuvers</th>
      <th>step_direction</th>
      <th>step_location_list</th>
      <th>speed</th>
      <th>pickup_pca0</th>
      <th>pickup_pca1</th>
      <th>dropoff_pca0</th>
      <th>dropoff_pca1</th>
      <th>pickup_group</th>
      <th>dropoff_group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-03-14 17:32:30</td>
      <td>40.765602</td>
      <td>-73.964630</td>
      <td>id2875421</td>
      <td>1</td>
      <td>2016-03-14 17:24:55</td>
      <td>40.767937</td>
      <td>-73.982155</td>
      <td>N</td>
      <td>455.0</td>
      <td>...</td>
      <td>depart|rotary|turn|new name|arrive</td>
      <td>left|straight|right|straight|arrive</td>
      <td>-73.982316,40.767869|-73.981997,40.767688|-73....</td>
      <td>12.183748</td>
      <td>0.007691</td>
      <td>0.017053</td>
      <td>-0.009666</td>
      <td>0.013695</td>
      <td>33</td>
      <td>19</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-06-12 00:54:38</td>
      <td>40.731152</td>
      <td>-73.999481</td>
      <td>id2377394</td>
      <td>1</td>
      <td>2016-06-12 00:43:35</td>
      <td>40.738564</td>
      <td>-73.980415</td>
      <td>N</td>
      <td>663.0</td>
      <td>...</td>
      <td>depart|turn|turn|end of road|continue|arrive</td>
      <td>none|right|left|right|left|arrive</td>
      <td>-73.980429,40.73857|-73.985444,40.731658|-73.9...</td>
      <td>7.569880</td>
      <td>0.007677</td>
      <td>-0.012371</td>
      <td>0.027145</td>
      <td>-0.018652</td>
      <td>7</td>
      <td>36</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-01-19 12:10:48</td>
      <td>40.710087</td>
      <td>-74.005333</td>
      <td>id3858529</td>
      <td>1</td>
      <td>2016-01-19 11:35:24</td>
      <td>40.763939</td>
      <td>-73.979027</td>
      <td>N</td>
      <td>2124.0</td>
      <td>...</td>
      <td>depart|turn|turn|turn|new name|turn|on ramp|me...</td>
      <td>right|left|right|left|straight|right|straight|...</td>
      <td>-73.978874,40.764148|-73.977685,40.763646|-73....</td>
      <td>14.409588</td>
      <td>0.004803</td>
      <td>0.012879</td>
      <td>0.034222</td>
      <td>-0.039337</td>
      <td>0</td>
      <td>18</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-04-06 19:39:40</td>
      <td>40.706718</td>
      <td>-74.012268</td>
      <td>id3504673</td>
      <td>1</td>
      <td>2016-04-06 19:32:31</td>
      <td>40.719971</td>
      <td>-74.010040</td>
      <td>N</td>
      <td>429.0</td>
      <td>...</td>
      <td>depart|turn|end of road|arrive</td>
      <td>left|left|right|arrive</td>
      <td>-74.010145,40.719982|-74.011527,40.714294|-74....</td>
      <td>7.546226</td>
      <td>0.038342</td>
      <td>-0.029194</td>
      <td>0.041343</td>
      <td>-0.042293</td>
      <td>78</td>
      <td>38</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-03-26 13:38:10</td>
      <td>40.782520</td>
      <td>-73.972923</td>
      <td>id2181028</td>
      <td>1</td>
      <td>2016-03-26 13:30:55</td>
      <td>40.793209</td>
      <td>-73.973053</td>
      <td>N</td>
      <td>435.0</td>
      <td>...</td>
      <td>depart|turn|turn|turn|arrive</td>
      <td>right|left|right|left|arrive</td>
      <td>-73.972998,40.793187|-73.976607,40.788361|-73....</td>
      <td>11.526767</td>
      <td>-0.002877</td>
      <td>0.041749</td>
      <td>-0.002380</td>
      <td>0.031071</td>
      <td>75</td>
      <td>69</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 61 columns</p>
</div>



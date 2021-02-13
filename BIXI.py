# -*- coding: utf-8 -*-
"""
Created on Fri May 22 20:06:50 2020

@author: torna
"""

#Bixi

data_urls = {
  "2014": "https://sitewebbixi.s3.amazonaws.com/uploads/docs/biximontrealrentals2014-f040e0.zip",
  "2015": "https://sitewebbixi.s3.amazonaws.com/uploads/docs/biximontrealrentals2015-69fdf0.zip",
  "2016": "https://sitewebbixi.s3.amazonaws.com/uploads/docs/biximontrealrentals2016-912f00.zip",
  "2017": "https://sitewebbixi.s3.amazonaws.com/uploads/docs/biximontrealrentals2017-d4d086.zip",
  "2018": "https://sitewebbixi.s3.amazonaws.com/uploads/docs/biximontrealrentals2018-96034e.zip",
  "2019": "https://sitewebbixi.s3.amazonaws.com/uploads/docs/biximontrealrentals2019-33ea73.zip",
}

import io
import pandas as pd
import requests
import zipfile
import calendar
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
df = None
for year, url in data_urls.items():
  print("Processing {}".format(year))
  # Load the url
  response = requests.get(url)
  # Read the archive from the response
  archive = zipfile.ZipFile(io.BytesIO(response.content))
  # Loop over all the files in the archive
  for file in archive.namelist():
    # Check that we are looking at one of the files we want
    if not archive.getinfo(file).is_dir() and "Station" not in file:
      print("Loading data from: {}".format(file))
      # We will load the start_date column only to save on memory use
      try:
        current_length = len(df)
        df = df.append(
          pd.read_csv(archive.open(file), usecols=["start_date"]),
          ignore_index=True, 
        )
      except:
        current_length = 0
        df = pd.read_csv(archive.open(file), usecols=["start_date"])
      print(" > {} rows processed".format(len(df) - current_length))
  response.close()

#Transform start_date column into panda format  
df["start_date"] = pd.to_datetime(df["start_date"]).dt.date
#Adding a date column that holds the dates of the original Df into manipulable datetime format
df["date"]=pd.to_datetime(df["start_date"], format="%Y/%m/%d")
#Adding a day column that modifies the date column but only takes the day (date is manipulable now)
df['day']=df['date'].dt.dayofyear
#Same manipulation to add months column playing on datetime
df["month"]=df["date"].dt.month
df.head()

target_df=df.groupby('date').date.size()



df['weekday']=df['date'].dt.day_name()

#Creates an array with all the days of the week (one for each day)
df['weekday'].unique()
#Creates a data frame with the sum of the datas for each days of the week
#Normalize gives the relative frequency among all datas (percentage)
day_df=df['weekday'].value_counts(normalize=True)
#Reorder day_df such that the days of the week are in order
day_df=day_df.reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
#print(day_df)
#day_df contains every day of the week and the relative frequency of bixi used during those days in 
#the last 5 years

# pandas has a built-in function to create dummy variable columns from categorical data
feature_df = pd.get_dummies(df.groupby("date").first(), columns=['weekday'], prefix="", prefix_sep="").loc[:,["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]]

#months column
for i, month_of_year in enumerate(feature_df.index.month_name().unique()):
    feature_df[month_of_year] = (feature_df.index.month == i+1).astype(int)
feature_df


#Feature matrix
features_matrix=feature_df.drop(columns=['Sunday', 'November'])
features_matrix



from matplotlib import dates as mdate


import math
from sklearn.metrics import mean_squared_error



# Montreal (Mirabel)
station_id = 48374

# Initialize df with 2012 data
meteo_df = pd.read_csv("https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID={}&Year={}&Day=14&timeframe=2&submit=Download+Data".format(station_id, 2014))
#meteo_hourly=pd.read_csv("https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID={}&Year=${}&Day=14&timeframe=1&submit=Download+Data".format(station_id,2014, header=None))

# Append other years
for year in range(2015, 2020):
  meteo_df = meteo_df.append(pd.read_csv("https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID={}&Year={}&Day=14&timeframe=2&submit=Download+Data".format(station_id, year)))


weather_features_df = meteo_df[["Date/Time", "Mean Temp (°C)", "Max Temp (°C)", "Min Temp (°C)", "Total Precip (mm)"]]



weather_features_df = weather_features_df.fillna(method="pad")


weather_features_df = weather_features_df.set_index("Date/Time")


features_matrix = features_matrix.join(weather_features_df[["Mean Temp (°C)", "Total Precip (mm)" ]])




import pandas as pd

raw_climat = pd.read_csv("https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID=48374&Year=2014&Month=1&Day=31&timeframe=1&submit=Download+Data")


for year in range (2014,2020):
  for month in range (1,13):
    if not (year == 2014 and month == 1):
      raw_climat = pd.concat([raw_climat, pd.read_csv("https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID=48374&Year=" 
                                            + str(year) + "&Month=" + str(month) + "&Day=31&timeframe=1&submit=Download+Data")]).reset_index(drop=True)

#print (raw_climat)


Trim = raw_climat[[ "Date/Time", "Temp (°C)", "Rel Hum (%)", "Wind Spd (km/h)", "Visibility (km)", "Wind Chill"]]
Trim.columns = ["date", "Average_temperature", "Average_humidity", "Average_wind_speed", "Average_visibility", "Average_wind_chill"]
Trim["date"] = pd.to_datetime(Trim["date"]).dt.date.values
Trim = Trim.groupby(["date"]).mean().reset_index()
Trim= Trim.fillna(method="pad")
Trim=Trim.set_index('date')
#print(Trim)


from sklearn import preprocessing

# Create x, where x the 'values of the weather' column's values as floats
Normal = Trim[["Average_temperature", "Average_humidity", "Average_wind_speed", "Average_visibility", "Average_wind_chill"]].values.astype(float)

# Create a minimum and maximum processor object
min_max_scaler = preprocessing.MinMaxScaler()

# Create an object to transform the data to fit minmax processor
Normal_scaled = min_max_scaler.fit_transform(Normal)

# Run the normalizer on the dataframe
Normalized_Trim = pd.DataFrame(Normal_scaled)


Normalized_Trim.columns = ['Average_temperature' , 'Average_humidity' ,  'Average_wind_speed' , 'Average_visibility' ,  'Average_wind_chill']
Normalized_Trim.index = Trim.index


#print(Normalized_Trim)


features_matrix = features_matrix.join(Normalized_Trim[["Average_temperature", "Average_humidity" , "Average_wind_speed" , "Average_humidity","Average_wind_chill", "Average_visibility"]])
features_matrix


#Taking Holidays into account
from dateutil.relativedelta import MO
from pandas.tseries.holiday import AbstractHolidayCalendar, GoodFriday, EasterMonday, Holiday, nearest_workday, next_monday, sunday_to_monday



class MontrealHolidayCalendar(AbstractHolidayCalendar):

  
  rules = [
    Holiday("New Year's Day", month=1, day=1, observance=next_monday),
    GoodFriday,
    EasterMonday,
    Holiday("Jour des Patriotes", month=5, day=24, offset=pd.DateOffset(weekday=MO(-1))),
    Holiday("St. Jean Baptiste", month=6, day=24, observance=nearest_workday),
    Holiday("Canada Day", month=7, day=1, observance=nearest_workday),
    Holiday("Labor Day", month=9, day=1, offset=pd.DateOffset(weekday=MO(1))),
    Holiday("Thanksgiving", month=10, day=1, offset=pd.DateOffset(weekday=MO(2))),
    Holiday("Christmas Day", month=12, day=25, observance=sunday_to_monday),
  ]  



MontrealHolidayCalendar().holidays(start="2014-01-01", end="2019-12-31")

features_matrix['holidays'] = features_matrix.index.isin(MontrealHolidayCalendar().holidays(start="2014-01-01", end="2019-12-31")).astype(int)

#Modelling
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model 
model.fit(features_matrix.loc[features_matrix.index.year < 2019], target_df.loc[target_df.index.year < 2019])
#print(model.coef_)
#print(model.intercept_)
params=pd.Series(model.coef_,index=features_matrix.columns)
error = mean_squared_error(model.predict(features_matrix.loc[features_matrix.index.year == 2019]), target_df.loc[target_df.index.year == 2019])
print(error) 

#Visualization
plt.figure(figsize=[21,9], dpi=300)
title_font = {
    'family': 'serif',
    'color':  'xkcd:raw umber',
    'weight': 'bold',
    'size': 22,
}
label_font = {
    'family': 'serif',
    'color':  'xkcd:black',
    'weight': 'normal',
    'size': 16,
}
plt.title(
    "Bixi Usage Rate Prediction For 2019", 
    fontdict=title_font,
    loc='left',
    pad=10
)
plt.xlabel(
    "Date",
    fontdict=label_font,
    labelpad=8
)
plt.ylabel(
    "Trips",
    fontdict=label_font,
    labelpad=8
    )
plt.annotate(
    "RMSE = {:.2f}".format(math.sqrt(error)),
    xy=(mdate.datestr2num("2019-01-06"),2500),
    fontsize=14,
    )
plt.plot(features_matrix.index, target_df, label='True') 
plt.plot(features_matrix.index, model.predict(features_matrix),label='Prediction')
plt.xlim(left=mdate.datestr2num("2019-01-01"), right=mdate.datestr2num("2019-12-31"))
plt.legend(loc='upper right')
plt.show()


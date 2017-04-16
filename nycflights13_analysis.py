
# coding: utf-8

# # Analysis of nycflights13 dataset using Python (Pandas, NumPy, and SkLearn)
# 
# nycflights13 contains details of all the flights that departed NYC in the year 2013.
# 
# Data  : https://cran.r-project.org/web/packages/nycflights13/nycflights13.pdf
# 
# 

# In[74]:

#IPython is what you are using now to run the notebook
import IPython
print ("IPython version:      %6.6s (need at least 1.0)" % IPython.__version__)

# Numpy is a library for working with Arrays
import numpy as np
print ("Numpy version:        %6.6s (need at least 1.7.1)" % np.__version__)

# SciPy implements many different numerical algorithms
import scipy as sp
print ("SciPy version:        %6.6s (need at least 0.12.0)" % sp.__version__)

# Pandas makes working with data tables easier
import pandas as pd
print ("Pandas version:       %6.6s (need at least 0.11.0)" % pd.__version__)

# Module for plotting
import matplotlib
print ("Mapltolib version:    %6.6s (need at least 1.2.1)" % matplotlib.__version__)

# SciKit Learn implements several Machine Learning algorithms
import sklearn
print ("Scikit-Learn version: %6.6s (need at least 0.13.1)" % sklearn.__version__)


# In[75]:

import matplotlib.pyplot as plt


# In[76]:

flights_df= pd.read_csv('flights.csv')
matplotlib.style.use('ggplot')


# In[77]:

print (flights_df.shape)
print (flights_df.columns)
print (flights_df.dtypes)


# In[78]:

flights_df.dest.unique()
flights_df.head(10)


# Let’s explore flights from NYC to Seattle. Using the flights dataset to answer the following questions.
# 
# (a) How many flights were there from NYC airports to Seattle in 2013?

# In[79]:

# Your code here
flights_nyc_seattle = flights_df[(flights_df['origin'].isin(['JFK', 'LGA', 'EWR'])) & 
           (flights_df['dest'] == 'SEA') & (flights_df['year'] == 2013)]
flights_nyc_seattle.shape[0]


# There were 3923 flights from NYC airport (JFK, LGA, EWR) to Seattle (SEA) in 2013.

# (b) How many airlines fly from NYC to Seattle?

# In[80]:

# Your code here
flights_nyc_seattle['carrier'].nunique()


# 
#  Five airlines fly from NYC to Seattle.

# (c) How many unique air planes fly from NYC to Seattle?

# In[201]:

# Your code here
flights_nyc_seattle['tailnum'].nunique()


#  935 unique air planes fly from NYC to Seattle.

# (d) What is the average arrival delay for flights from NC to Seattle?

# In[82]:

# Your code here
flights_nyc_seattle['arr_delay'].mean()


# The average arrival delay for flights from NYC to Seattle is -1.099 minutes. The negative time indicates that flights from NYC to Seattle arrived earlier than their usual arrival times.

# (e) What proportion of flights to Seattle come from each NYC airport?

# In[83]:

# Your code here
proportion_EWR_flights = flights_nyc_seattle[flights_nyc_seattle['origin'] == 'EWR'].shape[0]/flights_nyc_seattle.shape[0]
proportion_JFK_flights = flights_nyc_seattle[flights_nyc_seattle['origin'] == 'JFK'].shape[0]/flights_nyc_seattle.shape[0]
proportion_LGA_flights = flights_nyc_seattle[flights_nyc_seattle['origin'] == 'LGA'].shape[0]/flights_nyc_seattle.shape[0]
print('Proportion of flights from EWR = ' + str(proportion_EWR_flights * 100))
print('Proportion of flights from JFK = ' + str(proportion_JFK_flights * 100))
print('Proportion of flights from LGA = ' + str(proportion_LGA_flights * 100))


#  The proportion of flights from EWR to Seattle is 46.67%
#  
#  The proportion of flights from JFK to Seattle is 53.33%
#  
#  The proportion of flights from LGA to Seattle is 0.0%
#  

# 
# Flights are often delayed. Consider the following questions exploring delay patterns.
# 
# (a) Which date has the largest average departure delay? Which date has the largest average arrival delay?

# In[84]:

#Creating a new column date in the format YYYY/MM/DD.
flights_df['date'] = flights_df['year'].astype(str) + '/' + flights_df['month'].astype(str) + '/' + flights_df['day'].astype(str)
flight_avg_by_date = flights_df.groupby('date').mean()
print(flight_avg_by_date.sort_values(['dep_delay'], ascending=False)['dep_delay'].head(1))
print(flight_avg_by_date.sort_values(['arr_delay'], ascending=False)['arr_delay'].head(1))


# 
# 
# With an average departure delay of 83.53 minutes, NYC experienced the largest average departure delay of airlines on 8th March 2013.
# 
# With an average arrival delay of 85.86 minutes, NYC experienced the largest average arrival delay of airlines on 8th March 2013.

# (b) What was the worst day to fly out of NYC in 2013 if you dislike delayed flights?
# 

# In[85]:

# The average total delay can be one criteria to evaluate the worst day to fly out in 2013.
flight_avg_by_date['total_delay'] = flight_avg_by_date['arr_delay'] + flight_avg_by_date['dep_delay']
print(flight_avg_by_date.sort_values(['total_delay'], ascending=False)['total_delay'].head(1))


# Another criteria to evaluate the worst day can be the percentage of flights whose departure was delayed.
flights_dep_delay = flights_df.loc[flights_df['dep_delay'] > 0,:]
flights_dep_delay = pd.DataFrame(flights_dep_delay.groupby('date').count())
flights_dep_delay = flights_dep_delay.rename(columns={'Unnamed: 0' : 'count'})

flights_per_day = pd.DataFrame(flights_df.groupby('date').count())
flights_per_day = flights_per_day.rename(columns={'Unnamed: 0' : 'count'})
print((flights_dep_delay['count'] / flights_per_day['count'] * 100).sort_values(ascending=False).head(1))


# 1. If we consider the average total delay as a criteria, then 8th March 2013, with an average total delay of 169.40 minutes, was the worst day to fly out of NYC.
# 2. If we consider the percentage of flights whose departure was delayed as a criteria, then 23rd December 2013 was the worst day to flight out as 68.42% of the scheduled flights were delayed on that day.

# (c) Are there any seasonal patterns in departure delays for flights from NYC?

# In[86]:

# Plotting average departure delay per months in 2013.
plt.plot(flights_df.groupby('month').mean().dep_delay)
plt.xlabel('Months')
plt.ylabel('Average Departure Delay (minutes)')
plt.title('Average Departure Delay by Months')
plt.show()


#  
#  After analysing the average departure delay of flights by months in 2013, it can be observed that the average departure delay was highest in July followed by June and December. A major reason for the delay in flights could be the holiday season (summer and winter) where most of the people tend to fly.

# (d) On average, how do departure delays vary over the course of a day?

# In[87]:

# Plotting average departure delay by course of a day.
plt.bar(range(len(flights_df.groupby('hour').mean().dep_delay)), flights_df.groupby('hour').mean().dep_delay, 
        color = 'blue')
plt.xlabel('Hour of the day')
plt.ylabel('Average Departure Delay (minutes)')
plt.title('Average Departure Delay by Course of a day')
plt.show()


# Analysing total number of flights for each hour of the day.
plt.bar(range(len(flights_df.groupby('hour').count())), flights_df.groupby('hour').count()['Unnamed: 0'])
plt.xlabel('Hour of the day')
plt.ylabel('Total number of flights')
plt.title('Total number of flights by Course of a day')
plt.show()


# After analyzing the bar-chart (Average Departure Delay by Course of a day), it is evident that the average delay increases as the day progresses. A possible cause of this can be that delays in flights during the first half of the day cause more delays in flights during the latter half of the day. The hours from 1 a.m. to 4 a.m. can be considered as outliers as the total number of flights in this time range is very less compared to the number of flights at each of the hours from 5 a.m. to 12 a.m.

# ## Question 3
#     Which flight departing NYC in 2013 flew the fastest?

# In[88]:

#Calculating speed in miles per hour.
flights_df['speed'] = flights_df['distance']*60/flights_df['air_time']
columns = ['carrier', 'tailnum', 'flight', 'origin', 'dest','date', 'speed', 'distance', 'air_time']
flights_df.sort_values(['speed'], ascending=False)[columns].head(1)


# Delta Airlines (carrier DL) with a flight number of 1499 and tail number N666DN, travelling from LaGuardia Airport(LGA) airport to Hartsfield–Jackson Atlanta International Airport (ATL) was the flight that flew the fastest in 2013. It had a speed of 703.38 miles per hour.
#  

# ## Question 4
# Which flights (i.e. carrier + flight + dest) happen every day? Where do they fly to?

# In[107]:

# Your code here
flights_df['carrier_flight_dest'] = flights_df['carrier'].astype('str') + ' ' + flights_df['flight'].astype('str') + ' ' + flights_df['dest'].astype('str')

flights_by_carrier_flight_dest = flights_df.groupby('carrier_flight_dest').count()
flights_by_carrier_flight_dest = flights_by_carrier_flight_dest.rename(columns = {'Unnamed: 0' : 'count'})
flights_by_carrier_flight_dest = flights_by_carrier_flight_dest[flights_by_carrier_flight_dest['count'] == 365]
everyday_flights = flights_by_carrier_flight_dest.index.values.tolist()

destinations = []
for flight in everyday_flights:
    destination = flight[len(flight) - 3 : ]
    if(destination not in destinations):
        destinations.append(destination)

print('The flights that fly everyday are : ')
for flight in everyday_flights:
    print(flight)
    
print('The destinations that they fly to are : ')
for destination in destinations:
    print(destination)

## Question 5
Develop one research question you can address using the nycflights2013 dataset. Provide two visualizations to support your exploration of this question. Discuss what you find.

# In[143]:

# Finding the best airport amongst the 3 airports in NYC
dep_delay_by_origin = flights_df.groupby('origin').mean()['dep_delay']
postions = np.arange(len(dep_delay_by_origin))
plt.bar(postions, dep_delay_by_origin)
plt.xlabel('NYC Airports')
plt.ylabel('Average departure delay')
plt.xticks(postions, dep_delay_by_origin.index.values)
plt.title('Average Departure Delay for each NYC Airport')
plt.show()


delayed_flights = flights_df[flights_df['dep_delay'] > 0]
delayed_flights = delayed_flights.groupby('origin').count().dep_delay
postions = np.arange(len(delayed_flights))
plt.bar(postions, delayed_flights, color = 'blue')
plt.xlabel('NYC Airports')
plt.ylabel('Total number of delayed flights')
plt.xticks(postions, delayed_flights.index.values)
plt.title('Total number of delayed flights for each NYC Airport')
plt.show()


# Research Question: Which was the best NYC airport in 2013?
# 
# Criteria: The airport with the lowest average departure delay and the lowest number of delayed flights can be considered as the best airport in NYC in 2013.
# 
# Looking at both bar plots mentioned above, we can conclude that LaGuardia Airport (LGA), with an average departure delay of 10.34 minutes and 33690 delayed flights, was the best airport to fly out of NYC in the year 2013.
# 

# ## Question 6
# What weather conditions are associated with flight delays leaving NYC? Use graphics to explore.

# In[203]:

# Your code here
weather_df = pd.read_csv('weather.csv')
weather_df = weather_df.groupby(['year','month','day','origin']).mean()
weather_df.reset_index(level = 0, inplace = True)
weather_df.reset_index(level = 0, inplace = True)
weather_df.reset_index(level = 0, inplace = True)
weather_df.reset_index(level = 0, inplace = True)
weather_df = weather_df[['origin','year','month','day','temp','dewp','humid','wind_dir','wind_speed','wind_gust','precip','pressure','visib']]
weather_df.head()

flights_agg_mean = flights_df.groupby(['year','month','day','origin']).mean()
flights_agg_mean.reset_index(level = 0, inplace = True)
flights_agg_mean.reset_index(level = 0, inplace = True)
flights_agg_mean.reset_index(level = 0, inplace = True)
flights_agg_mean.reset_index(level = 0, inplace = True)
flights_agg_mean.head()

flights_weather_joined = pd.merge(flights_agg_mean, weather_df, on=['year','month','day','origin'], how='inner')
flights_weather_joined['precip'] = flights_weather_joined['precip'] * 1000;
print(flights_weather_joined.head())


# In[190]:

import seaborn as sns
get_ipython().magic('matplotlib inline')

sns.pairplot(flights_weather_joined, x_vars = ['temp','dewp','humid'], y_vars = 'dep_delay', size = 7,
             aspect = 0.7, kind='reg')
sns.pairplot(flights_weather_joined, x_vars = ['wind_dir', 'wind_speed','wind_gust'], y_vars = 'dep_delay', size = 7,
             aspect = 0.7, kind='reg')
sns.pairplot(flights_weather_joined, x_vars = ['precip', 'pressure','visib'], y_vars = 'dep_delay', size = 7,
             aspect = 0.7, kind='reg')


# After examining the scatter-plots of the departure delays of flights with respect to all nine weather related factors (temperature, dewpoint, humidity, wind direction, wind speed, wind gust, precipitation, pressure, and visibility), the following conclusions can be made:  
# 1. There seems to be a fairly strong positive correlation between humidity and departure delay.
# 2. There seems to be a fairly strong positive correlation between precipitation and departure delay.
# 3. There seems to be a fairly weak positive correlation between temperature (air) and departure delay.
# 4. There seems to be a fairly strong negative correlation between visibility and departure delay.
# 
# Linear Regression can be used to affirm the findings from the scatter plots.
# Performing a multiple linear regression to check the correlation between departure delay and the four factors including humidity, precipitation, temperature, and visibility. 

# In[194]:

from sklearn.linear_model import LinearRegression
x = flights_weather_joined[['temp','precip','humid','visib']]
y = flights_weather_joined['dep_delay']
linreg = LinearRegression()
linreg.fit(x, y)
print (linreg.intercept_)
print (linreg.coef_)
list(zip(['temp','precip','humid','visib'], linreg.coef_))


# The linear regression equation is as follows: 
# 
# y (Departure Delay) = 16.18 + 0.05 x Temperature + 0.18 x Precipitation + 0.19 x Humidity - 1.80 x Visibility
# 
# The interpretation is as follows:
# 
# For a given value of precipitation, humidity, visibility, a unit increase in temperature increases the departure delay by 0.05 minutes.
# 
# For a given value of temperature, humidity, visibility, a unit increase in precipitation increases the departure delay by 0.18 minutes.
# 
# For a given value of precipitation, humidity, temperature, a unit increase in visibility decreases the departure delay by 1.80 minutes.
# 
# For a given value of precipitation, temperature, visibility, a unit increase in humidity increases the departure delay by 0.19 minutes.

# In[ ]:




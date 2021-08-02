# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 20:27:34 2021
Stock Market Clustering With Python
@author: PMR

This code retrieves information related to companies from the data base of Yahoo Finance
and creates clusters by divinding data points into a number of groups

"""

# %% Import libraries

import pandas_datareader as web
import pandas as pd
import numpy as np
import datetime as dt
from yahoo_fin import stock_info as si

from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans


# %% Setup analysis and import data

# downloads list of tickers (e.g. S&P 500)
companies = si.tickers_sp500()

#number of clusters
cluster = 5

# time period
start = dt.datetime.now() - dt.timedelta(days=365*2)
end = dt.datetime.now()

# import data from online sources (e.g. Yahoo Finance)
data = web.DataReader(list(companies), 'yahoo', start, end)

# choose info to cluster
open_values = np.array(data['Open'].T)
close_values = np.array(data['Close'].T)
daily_movements = close_values - open_values


# %% Run clustering

# scale input vectors individually to unit norm
normalizer = Normalizer()

# K-Means clustering
clustering_model = KMeans(n_clusters=cluster, max_iter=1000)

# construct pipeline to automate machine learning workflows
pipeline = make_pipeline(normalizer, clustering_model)

# fit all the transforms and transform the data, then fit the transformed data using the final estimator
pipeline.fit(daily_movements)

# apply transforms to the data, and predict with the final estimator
clusters = pipeline.predict(daily_movements)

# print results
results = pd.DataFrame({
    'clusters': clusters,
    'tickers': list(companies)
    }).sort_values(by=['clusters'], axis=0)

print(results)



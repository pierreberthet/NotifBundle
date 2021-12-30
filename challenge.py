#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 19:03:09 2021

@author: pierre
"""

import os
import numpy as np
import pandas as pd
# import modin.pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm
from datetime import datetime, timedelta
import requests

# import ray
# ray.init()

import argparse

parser = argparse.ArgumentParser(description='DataScience from URL')
parser.add_argument('URL', metavar='URL', type=str, nargs='+',
                    help='URL of the .csv file')
parser.add_argument('output_path', metavar='SaveFile', type=str, nargs='+',
                    help='Path to save the output .csv file')
args = parser.parse_args()


#%%
############################################
#
#           INPUT
#
############################################

# check if dir path exist, otherwise create it
if os.path.split(args.output_path[0])[0] != '' and not os.path.isdir(os.path.split(args.output_path[0])[0]):
   os.makedirs(os.path.split(args.output_path[0])[0])

# load csv file from url to DataFrame
# url = 'https://static-eu-komoot.s3.amazonaws.com/backend/challenge/notifications.csv'
# df = pd.read_csv(
#     url, names=['timestamp', 'user_id', 'friend_id', 'friend_name'])
df = pd.read_csv(
    args.URL[0], names=['timestamp', 'user_id', 'friend_id', 'friend_name'])

# enhance dataframe for ease of use
df['timestamp'] = pd.to_datetime(df.timestamp)
df = df.sort_values(by='timestamp', axis=0, ignore_index=True)  # we make sure the data is sorted chronologically
df['dayofyear'] = [row.dayofyear for row in df.timestamp]  # easier to use than timestamp/datetime
df['hours'] = [row.hour for row in df.timestamp]  # for inter/intra day analyses
df['base'] = [row.hour * 60 + row.minute for row in df.timestamp]


#%%
############################################
#
#           FUNCTIONS
#
############################################

def get_one_error(checkpoints, timepoint, metric='hour'):
    # returns error of 1 timepoint
    assert metric == 'hour'  or metric == 'minute', f"please use only minute or hour as metric"
    checkpoints = np.array(sorted(checkpoints))
    if metric == 'hour':
        day_v = 24
    else:
        day_v = 24 *60
    # print(day_v)
    if timepoint > checkpoints[-1]:
        return checkpoints[0] + (day_v - timepoint)
    else:
        return  checkpoints[np.argwhere(checkpoints >= timepoint)[0][0]] - timepoint
    

def linear_loss(checkpoints: list, timepoints, metric='hour'):
    # linear loss function
    assert metric == 'hour'  or metric == 'minute', f"please use only minute or hour as metric"
    checkpoints = np.array(sorted(checkpoints))
    error = 0
    for tp in timepoints:
        temp = get_one_error(checkpoints, tp, metric)
        if temp < 0:
            print(f"{temp} checkpoints {checkpoints} {tp}")
        error += temp
    return error / len(timepoints)


def get_push_batch(timedata:pd.Series, checkpoints:list):
    # assign a checkpoint batch to timepoints
    batch = np.ones(timedata.shape[0]) * checkpoints[0]
    for bx, checkpoint in enumerate(checkpoints):
        if bx < len(checkpoints) -1:
            batch[np.argwhere(timedata.values >= checkpoint)] = checkpoints[bx + 1]
        else:
            batch[np.argwhere(timedata.values >= checkpoint)] = np.max(timedata) + 1 # arbitrary number to signal this will be dealt by the following day notification.
    return batch


def get_tours_message(df:pd.DataFrame, day:int, user_id:str, checkpoint:int or float, batch_version='batch_kpp', display=True):
    # bundle up notification per user and by day, by checkpoints
    res = pd.DataFrame(columns=['notification_sent', 'timestamp_first_tour', 'tours', 'receiver_id', 'message'], index=range(4))
    limit = df[batch_version].max()
    for bx, batch in enumerate(df[batch_version].unique()):
        if batch != limit:
            if batch == df[batch_version].min() and day > df.dayofyear.min():
                uday = df.query("(user_id == @user_id and dayofyear == (@day-1) and batch_kpp == @limit) or (user_id == @user_id and dayofyear == @day and batch_kpp == @batch)")
            else:
                uday = df.query("user_id == @user_id and dayofyear == @day and batch_kpp == @batch")
            if uday.shape[0] > 0:
                n_friend = uday.friend_id.nunique() - int( user_id in uday.friend_id.unique())
                friend_name = uday.friend_name.iloc[0]
                if n_friend > 0:
                    if n_friend > 1:  # what about when same friend(s) had several tours during the same time interval?
                        if uday.friend_id.iloc[0] == user_id:
                            friend_name = uday.friend_name.iloc[1]
                        if n_friend > 2:
                            msg = f"{friend_name} and {n_friend - 1} others went on a tour"
                        else:
                            msg = f"{friend_name} and 1 other went on a tour"
                    else: 
                        if uday.friend_id.iloc[0] == user_id:
                            friend_name = uday.friend_name.iloc[1]
                        msg = f"{friend_name} went on a tour"
                    res.loc[bx] = [pd.Timestamp(uday.timestamp.max().year, uday.timestamp.max().month, uday.timestamp.max().day,
                                                int(batch // 60), int(batch % 60)), 
                                   uday.timestamp.min(), uday.shape[0], user_id, msg]
                    if display:                        
                        print("{},{},{},{},{}".format(pd.Timestamp(uday.timestamp.max().year, uday.timestamp.max().month, uday.timestamp.max().day,
                                                     int(batch // 60), int(batch % 60)), 
                                        uday.timestamp.min(), uday.shape[0], user_id, msg))
    return res.dropna(axis=0)



###
###         BRUTE FORCE ON HOURS
###

# for r, row in df.iterrows():
#     print(row[['hours', 'batch']])
# from itertools import combinations

# min_error = 24
# combi = combinations(np.arange(0, 24, 3), 4)
# for c in tqdm(combi):
#     error = linear_loss(list(c), df.hours.values) 
#     if error < min_error:
#         min_error = error
#         points = list(c)
# combi = combinations(np.arange(1, 24, 3), 4)
# for c in tqdm(combi):
#     error = linear_loss(list(c), df.hours.values) 
#     if error < min_error:
#         min_error = error
#         points = list(c)
# combi = combinations(np.arange(2, 24, 3), 4)
# for c in tqdm(combi):
#     error = linear_loss(list(c), df.hours.values) 
#     if error < min_error:
#         min_error = error
#         points = list(c)





######
######     K-Means Clustering
######

kmeans = KMeans(init="random",
                n_clusters=4,
                n_init=10,
                max_iter=300)

kmeans.fit(np.reshape(df.base.values, [len(df.base.values), 1]))
kp = np.array(sorted(kmeans.cluster_centers_.flatten()))
df['km_labels'] = kmeans.labels_

######
######     K-Means Improvements
######

# we move the means to the inter checkpoints average,
# or end of the day for the last checkpoint,
# to account for the one sided ownership of timepoints to checkpoints.
# slight improvement on the error perf compared ot the initial K-means values.

kp2 = np.zeros(len(kp))
for i, checkpoint in enumerate(kp):
   if i < len(kp) - 1:
       kp2[i] = (kp[i] + kp[i + 1]) //2
   else:
       kp2[i] = (kp[i] + 1440) //2

df['batch_kpp'] = get_push_batch(df.base, kp2)






############################################
#
#           OUTPUT
#
############################################

# initialise output dataframe
out = pd.DataFrame(columns=['notification_sent', 'timestamp_first_tour', 'tours', 'receiver_id', 'message'])

# print columns names to stdout
head = ''
for col in out.columns.values:
    if col != out.columns.values[-1]:
        head += f"{col},"
    else:
        head += f"{col}"
print(head)

# call the bundling functions per user and per day, accumulate in a dataframe and print results to stdout
for day in df.dayofyear.unique():
    for user_id in df.query("dayofyear == @day").user_id.unique():
        out = out.append(get_tours_message(df, day, user_id, kp2))
        
out.reset_index(drop=True).to_csv(args.output_path[0], index=False, sep=',')




#%%
############################################
#
#           FIGURES
#
############################################

from matplotlib import pyplot as plt
import seaborn as sns

# Seaborn parameters
cmap = sns.color_palette("colorblind")
sns.set_palette(cmap)
sns.set_style('darkgrid')
sns.set_context('notebook')



# Countplot of events for each day
f, ax = plt.subplots(figsize=(12, 9))
ax.hist(df.dayofyear.values, bins=df.dayofyear.nunique())
ax.set_xlabel('day of the year')
ax.set_ylabel('# events')

# Distribution of events by hour of the day
f, ax = plt.subplots(figsize=(12, 9))
ax.hist(df.hours, bins=24)
ax.set_xlabel('hour of the day')
ax.set_ylabel('# events')

# Distribution of events by minutes of the day
f, ax = plt.subplots(figsize=(12, 9))
ax.hist(df.base.values, bins=100)
ax.set_xlabel('minutes of the day')
ax.set_ylabel('# events')


# Histogram of the events by hour of the day, color coded by checkpoint relevance.
f, ax = plt.subplots(figsize=(12, 9))
sns.histplot(data=df, x='hours', hue='batch', palette=sns.color_palette("Spectral", as_cmap=True), ax=ax)

# Histogram of the events by hour of the day, color coded by checkpoint relevance.
f, ax = plt.subplots(figsize=(12, 9))
sns.histplot(data=df, x='base', hue='batch_kpp', palette=sns.color_palette("Spectral", as_cmap=True), ax=ax)
ax.set_title('Color coded checkpoint ownership of the events by minutes of the day')
ax.set_xlabel('minute of the day')


#%%


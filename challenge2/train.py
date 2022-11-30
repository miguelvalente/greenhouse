   
from datetime import datetime, timedelta
import sys
sys.path.append('../') 

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xlrd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import Resources
from model import ContributionRNN

greenhouse = pd.read_parquet(paths.greenhouse)
production = pd.read_parquet(paths.production)
resources = pd.read_parquet(paths.resources)
weather = pd.read_parquet(paths.weather)

def print_deets(df):
    print(df.dtypes)
    print(df.isna().sum())
    print(df.head(-1))

def nan_details(df, treshold = 0):
    nans = list(df.isnull().sum())
    nan_cols = [ (col, nans[idx]) for idx, col in enumerate(df.columns) if nans[idx] > treshold]
    print(*nan_cols, sep='\n')
    to_drop = [col for col, nan in nan_cols]
    return to_drop

#Matches string values of Nan and replaces with np.nan
df = greenhouse
df = df.replace(['^ +NaN'], np.nan, regex=True)
# Shows collums filled almost halfway with Nans and drops them
to_drop = nan_details(df, len(df)*0.4)
df.drop(to_drop, axis=1, inplace=True)
#Prints remainder collums with Nans to be manually inspected
to_drop = nan_details(df)
# Drops all rows with Nans and converts remaidner to numeric
df.dropna(inplace=True)
df = df.apply(pd.to_numeric)
nan_details(df)

# Removes before normalizing to ensure no Nans are introduced
df.drop(columns=df.columns[df.nunique()==1], inplace=True)
# Min Max normalized collums and removes uniqye valued collums
# normalized_df = (df-df.min())/(df.max()-df.min())
# nan_details(normalized_df)
df['time'] = [xlrd.xldate_as_datetime(time, 0 ) for time in df['%time']]
df.drop(['%time'], axis=1, inplace=True)

df_days_unsum = {}
for day in df.groupby(by=[df.time.dt.day, df.time.dt.month]):
    key = pd.to_datetime(day[1].time.values[0]).date()
    df_days_unsum[key] = day[1]

dates = [key for key, _ in df_days_unsum.items()]

resources['time']= [xlrd.xldate_as_datetime(time, 0 ) for time in resources['%Time '].values]
resources.drop('%Time ',axis=1, inplace=True)

resources['time'] = resources['time'] + timedelta(days=1)
resources = resources.set_index('time')
resources['elec'] = resources['ElecHigh'] + resources['ElecLow']
resources = resources[['elec']]

for date in resources.index.date:
    if date not in df_days_unsum.keys():
        resources.drop(date, inplace=True)

Y_train, Y_test = train_test_split(resources, test_size=0.2, random_state=1337)
X_train = { date: df_days_unsum[date].drop('time',1) for date in Y_train.index.date}
X_test = { date: df_days_unsum[date].drop('time',1) for date in Y_test.index.date}
std, mean = [], []
for _, value in X_train.items():
    std.append(value.values)
    mean.append(value.values)

std = np.vstack(np.array(std)).std(axis=0)
mean = np.vstack(np.array(mean)).mean(axis=0)
for key, value in X_train.items():
    X_train[key] = value - mean / std
for key, value in X_test.items():
    X_test[key] = value - mean / std

epochs = 10
batch_size = 1

train_loader = DataLoader(Resources(X_train, Y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(Resources(X_test, Y_test), batch_size=batch_size, shuffle=True)

model = ContributionRNN(40, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = nn.MSELoss()

history = {'mse':  [], 'mse_val': [], 'loss': [], 'loss_val':[]}
for epoch in trange(epochs):
    model.train()
    y_pred = []
    y_ground = []
    losses = []
    for x, y in train_loader:
        optimizer.zero_grad()
        output = model(x.squeeze().float())
        loss = loss_fn(output.squeeze(), y.float())
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        y_pred.append(output.detach().numpy().squeeze())
        y_ground.append(y.numpy().squeeze())


    history['loss'].append(sum(losses)/len(losses))
    history['mse'].append(mean_squared_error((y_ground, y_pred)))

    model.eval()
    y_pred = []
    y_ground = []
    losses = []
    for x, y in val_loader:
        with torch.no_grad():
            output = model(x.squeeze().float())
            loss = loss_fn(output.squeeze(), y.float())

            losses.append(loss.item())
            y_pred.append(output.detach().numpy().squeeze())
            y_ground.append(y.numpy().squeeze())

    history['loss_val'].append(sum(losses)/len(losses))
    history['mse_val'].append(mean_squared_error((y_ground, y_pred)))

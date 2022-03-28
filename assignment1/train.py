   
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm, trange

from dataset import Weather
from model import Lights

greenhouse = pd.read_parquet(r'challenge/GreenhouseClimate.parquet')
production = pd.read_parquet(r'challenge/Production.parquet')
resources = pd.read_parquet(r'challenge/Resources.parquet')
weather = pd.read_parquet(r'challenge/Weather.parquet')
# Print datatypes
print(weather.dtypes)

# Describe columns
weather.describe(include='all')
# Preview the dataset
weather.head(-1)
# Check if there are null values in any of the columns. You will see `Unnamed: 32` has a lot.
weather.isna().sum()
df = pd.concat([weather,greenhouse['AssimLight']], axis = 1)
df.dropna(inplace=True)
df.drop('%time', axis=1, inplace=True)
df["AssimLight"] = pd.to_numeric(df["AssimLight"])
df.head(-1)
# Get rid of sklearn errors
map_dict = {100: 1}
df['AssimLight'].replace(map_dict,inplace=True)

X = df.drop('AssimLight', 1)
Y = df["AssimLight"]



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state = 1337)

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

epochs = 10
batch_size = 16

train_loader = DataLoader(Weather(X_train_scaled, Y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(Weather(X_test_scaled, Y_test), batch_size=batch_size, shuffle=True)

model = Lights()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = nn.BCEWithLogitsLoss()

history = {'acc':  [], 'acc_val': [], 'loss': [], 'loss_val':[]}
for epoch in trange(epochs):
    model.train()
    y_pred = []
    y_ground = []
    losses = []
    for x, y in train_loader:
        optimizer.zero_grad()
        output = model(x.float())
        loss = loss_fn(output.squeeze(), y.float())
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        y_pred.append(output.detach().numpy())
        y_ground.append(y.numpy())

    # acc = accura.float().squezeeze()).float()).float().squezeeze()).float()).float().squezeeze()).float()).float().squezeeze()).float())cy_score(Y_test, y_predict_r)
    y_pred = np.where(np.concatenate(y_pred) > 0.5, 1, 0)
    y_ground = np.concatenate(y_ground)
    acc = sum(y_pred == y_ground) / len(y_pred)

    history['loss'].append(sum(losses)/len(losses))
    history['acc'].append(acc)

    model.eval()
    y_pred = []
    y_ground = []
    losses = []
    for x, y in val_loader:
        with torch.no_grad():
            output = model(x.float())
            loss = loss_fn(output.squeeze(), y.float())

            losses.append(loss.item())
            y_pred.append(output.detach().numpy())
            y_ground.append(y.numpy())

    y_pred = np.where(np.concatenate(y_pred) > 0.5, 1, 0)
    y_ground = np.concatenate(y_ground)
    acc = sum(y_pred == y_ground) / len(y_pred)

    history['loss_val'].append(sum(losses)/len(losses))
    history['acc_val'].append(acc)
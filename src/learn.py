import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

scaler_m = MinMaxScaler()

cos = np.array([])
date = np.array([])
label = np.array([])


# JSONファイルをPythonの辞書に変換します。
with open('../data/ans.json') as f:
    docs = json.load(f)
for doc in docs['true']:
    cos = np.append(cos, doc['cos'])
    date = np.append(date, doc['date'])
    label = np.append(label, 0)
for doc in docs['false']:
    cos = np.append(cos, doc['cos'])
    date = np.append(date, doc['date'])
    label = np.append(label, 1)

df = pd.DataFrame(data={'cos': cos, 'date': date, 'label': label})

mean = df['date'].mean()
std = df['date'].std()

print(f"mean:{mean}")
print(f"std:{std}")

df['date_std'] = (df[['date']] - mean)/std

label_categorical = to_categorical(df[['label']])

model = Sequential()
model.add(Dense(10, input_dim=2, activation='relu'))  # 入力層と隠れ層
model.add(Dense(2, activation='softmax'))  # 出力層

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

df = pd.DataFrame(data={'cos': df['cos'], 'date': df['date_std']})
model.fit(df, label_categorical, epochs=100, batch_size=10)

model.save('../data/trained_data')

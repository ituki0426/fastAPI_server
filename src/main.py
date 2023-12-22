from fastapi import FastAPI
import numpy as np

from keras.models import load_model

mean = 117.5364705882353
std = 52.3410265358211
model = load_model('../data/trained_data')
model.summary()

new_data = np.array([[0.9, 0.3], [0.4, 0.5]])
predictions = model.predict(new_data)

# 確率を出力
print(predictions)

app = FastAPI()

@app.get("/")
async def root(cos: float, date: int):
    date_std = (date - mean)/std
    predictions = model.predict(np.array([[cos, date_std]]))
    return {"message": f"query1:{cos}, query2:{date}"}

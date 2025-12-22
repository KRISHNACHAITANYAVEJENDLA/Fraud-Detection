import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

def predict_cover(features):
    features = np.array(features).reshape(1, -1)
    return int(model.predict(features)[0])

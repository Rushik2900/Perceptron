from  utils.all_units import save_plot
from utils.model import Perceptron
from utils.all_units import prepare_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib 
from matplotlib.colors import ListedColormap

plt.style.use("fivethirtyeight")

AND= {
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y" :[0,0,0,1]
}

df=pd.DataFrame(AND)


X,y = prepare_data(df)

ETA = 0.3

EPOCHS = 10

model=Perceptron(eta=ETA, epochs=EPOCHS)

model.fit(X, y)

_ =model.total_loss()

save_model(model, filename="and.model")
save_plot(df, "and.png")
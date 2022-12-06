from utils.all_units import save_plot
from  utils.all_units import save_model
from utils.model import Perceptron
from utils.all_units import prepare_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib 
from matplotlib.colors import ListedColormap

plt.style.use("fivethirtyeight")
def main(data, ETA, EPOCHS, filename, plotName):

    df=pd.DataFrame(data)


    X,y = prepare_data(df)

    model=Perceptron(eta=ETA, epochs=EPOCHS)

    model.fit(X, y)

    _ =model.total_loss()

    save_model(model, filename)
    save_plot(df, plotName, model)

if __name__=='__main__':
    AND= {
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y" :[0,0,0,1]
    }
    ETA = 0.3
    EPOCHS = 10
    main(AND, ETA, EPOCHS, filename="and.model", plotName="and.png")
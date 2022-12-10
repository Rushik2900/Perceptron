from utils.all_units import save_plot
from  utils.all_units import save_model
from utils.model import Perceptron
from utils.all_units import prepare_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib 
from matplotlib.colors import ListedColormap
import logging
import os

logging_str = "[%(asctime)s: %(filename)s:%(funcName)s:%(lineno)d] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"and.logs"), level=logging.INFO, format=logging_str, filemode="a")

plt.style.use("fivethirtyeight")

def main(data, ETA, EPOCHS, filename, plotName):

    df=pd.DataFrame(data)
    logging.info(f"This is actual dataFrame {df}")
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
    EPOCHS = 10000
    try:
        logging.info("\n>>>>> Starting training >>>>>>>>")
        main(AND, ETA, EPOCHS, filename="and.model", plotName="and.png")
        logging.info(">>>>> training done successfully <<<<<<")
    except Exception as e:
        logging.exception(e)
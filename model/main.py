import numpy as np
from model_classification_3D import *
import random
from tensorflow.keras import utils as np_utils

if __name__ == "__main__":
    dataNum = 50
    # simulation data
    newtrainXs = np.random.rand(dataNum, 40, 40, 40, 3)
    newtrainYs = [0] * dataNum
    newtrainYs = np.array(newtrainYs)
    # simulation label
    newtrainYs[0:int(len(newtrainYs) * 0.3)] = 1
    newtrainYs_cla = np_utils.to_categorical(newtrainYs)

    base_network = get_classification_models_network_3D_trip()
    
    history = base_network.fit(
        x = newtrainXs,
        y = newtrainYs,
        validation_split=0.2,
        epochs=10,
        batch_size=6,
        verbose=1,
        shuffle=True,
        )
    
    step2_network = getStep2TrainModel(base_network)

    history = step2_network.fit(
        x = newtrainXs,
        y = newtrainYs_cla,
        validation_split=0.2,
        epochs=10,
        batch_size=6,
        verbose=1,
        shuffle=True,
        )
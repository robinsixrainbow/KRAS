import numpy as np
from model_classification_3D import *
import random
from tensorflow.keras import utils as np_utils
import sys
from BatchBalance import *

if __name__ == "__main__":
    dataNum = 50

    
    useBatchBalance = 'useBatchBalance' in sys.argv
    useMetricModel  = 'useMetricModel' in sys.argv

    
    # 模擬3D資料
    newtrainXs = np.random.rand(dataNum, 40, 40, 40, 3)
    newtrainYs = [0] * dataNum
    newtrainYs = np.array(newtrainYs)
    # 模擬標籤
    newtrainYs[0:int(len(newtrainYs) * 0.3)] = 1
    # 模擬檔案名稱
    fileNames = [str(i)+'.mat' for i in range(dataNum)]

    # 是否使用 BatchBalance
    if useBatchBalance:
        print("Use BatchBalance")
        newtrainXs, newtrainYs, newfnames = balanceData(newtrainXs, newtrainYs, fileNames)
    else:
        print("No BatchBalance")

    newtrainYs_cla = np_utils.to_categorical(newtrainYs)
    base_network = get_classification_models_network_3D_trip(useMetricModel=useMetricModel)
    if useMetricModel:
        print("Use MetricModel")
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
    else:
        print("No MetricModel")
        history = base_network.fit(
            x = newtrainXs,
            y = newtrainYs_cla,
            validation_split=0.2,
            epochs=10,
            batch_size=6,
            verbose=1,
            shuffle=True,
            )
        
import datetime
import numpy as np
from BatchBlance import *

if __name__ == "__main__":
    dataNum = 10
    # simulation data
    xi = np.random.rand(dataNum, 40, 40, 40, 1)
    yi = [0] * dataNum
    yi = np.array(yi)
    # simulation label(unbalanced)
    yi[0:int(len(yi) * 0.3)] = 1
    # simulation filename
    fileNames = [str(i)+'.mat' for i in range(dataNum)]
    
    print("yi = ", yi)

    
    newXs, newYs, newfnames = blanceData(xi, yi, fileNames)
    print("newYs = ", newYs)
    print("newfnames = ", newfnames)
    
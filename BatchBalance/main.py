import datetime
import numpy as np
from BatchBalance import *

if __name__ == "__main__":
    dataNum = 10
    # 模擬3D資料
    xi = np.random.rand(dataNum, 40, 40, 40, 1)
    yi = [0] * dataNum
    yi = np.array(yi)
    # 模擬標籤不平衡狀況
    yi[0:int(len(yi) * 0.3)] = 1
    # 模擬檔案名稱
    fileNames = [str(i)+'.mat' for i in range(dataNum)]
    
    print("yi = ", yi)

    
    newXs, newYs, newfnames = balanceData(xi, yi, fileNames)
    print("newYs = ", newYs)
    print("newfnames = ", newfnames)
    
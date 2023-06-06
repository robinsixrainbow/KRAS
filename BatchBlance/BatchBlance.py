import numpy as np

# 平衡資料
def blanceData(dataXs, dataYs, fnames):
    # 預設2分類，dataYs為分類代號
    idx0 = np.where(dataYs == 0)[0]
    idx1 = np.where(dataYs == 1)[0]
    np.random.shuffle(idx0)
    np.random.shuffle(idx1)
    maxLen = max(len(idx0), len(idx1))
    newdataXs = []
    newdataYs = []
    newfnames = []

    for i in range(maxLen):
        idx0L = len(idx0)
        idx1L = len(idx1)

        idx0idx = i % idx0L
        idx1idx = i % idx1L

        n0idx = idx0[idx0idx]
        n1idx = idx1[idx1idx]

        newdataXs.append(dataXs[n0idx])
        newdataXs.append(dataXs[n1idx])

        newdataYs.append(int(dataYs[n0idx]))
        newdataYs.append(int(dataYs[n1idx]))

        newfnames.append(fnames[n0idx])
        newfnames.append(fnames[n1idx])

    newdataXs = np.array(newdataXs)
    newdataYs = np.array(newdataYs)


    return newdataXs, newdataYs, newfnames
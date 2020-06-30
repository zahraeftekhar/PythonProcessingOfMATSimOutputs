import pandas as pd
from shapely.geometry import Point
import time
from tqdm import tqdm
from time import sleep
from math import floor
import numpy as np
snapData = pd.read_csv("D:/ax/gis/completePLUdata_30sec/completePLUdata_30sec_0.csv")
interval = 30 #todo: please enter the interval between snapData records
snapDataSplit = snapData
snapDataSplit[['mzr_id', 'VEHICLE']] = snapDataSplit[['mzr_id', 'VEHICLE']].astype(int)
snapDataSplit = snapDataSplit.sort_values(by=["VEHICLE","TIME"])
vehicleIDs = snapDataSplit.VEHICLE.unique()
clusterData = pd.DataFrame()
i=0
startTime = time.time()
mm=range(0,3200,100)
for i in (range(len(vehicleIDs)-1)):#len(vehicleIDs) #1962 sec for one file of 2300 user
    for m in mm:
        if i==m:
            print('{percentage} percent____{duration} sec'.format(percentage = i/len(vehicleIDs), duration = time.time()-startTime))
    # if i==1:
    #     i
    testRecords = snapDataSplit[snapDataSplit.VEHICLE == vehicleIDs[i]]
    snapDataSplit = snapDataSplit.drop(index=range(len(testRecords)))
    snapDataSplit = snapDataSplit.reset_index(drop=True)
    clusIndex = [0]
    j = 0
    k = 1
    while k < len(testRecords):
        while testRecords['EASTING'][j]==testRecords['EASTING'][k]:
            if (testRecords['NORTHING'][j]==testRecords['NORTHING'][k]):
                k+=1
                if(k == len(testRecords)):
                    break
        # if((k-j)>1):
        clusIndex += [k]
        j=k
        k+=1
    newD = pd.DataFrame(index=range(len(clusIndex)-1), columns=testRecords.columns)
    newD ['duration(sec)'] = None
    l=0
    for l in range(len(clusIndex)-1):
        # if l == 88:
        #     l
        if l==len(clusIndex)-2:
            newD.iloc[l, -1] = testRecords.TIME[clusIndex[l + 1]-1] - testRecords.TIME[clusIndex[l]]+interval
        else:
            newD.iloc[l,-1] = testRecords.TIME[clusIndex[l+1]]-testRecords.TIME[clusIndex[l]]
        newD.iloc[l,0:-1]=testRecords.iloc[clusIndex[l],:]
    clusterData = clusterData.append(newD)
print(time.time() - startTime)
clusterData.to_csv('D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/clusterPLU_30sec_0.CSV', index=False, header=True)

#_______________ location type identification _____________________________________________________



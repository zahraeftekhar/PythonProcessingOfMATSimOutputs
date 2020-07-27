import pandas as pd
from shapely.geometry import Point
import time
from math import floor
import numpy as np
# from groundTruthAnalysis_homeWorkDistribution import forbiddenIDs


startTime = time.time()


# matsimOutput= pd.read_csv('D:/ax/gis/MATSimOutput_snapshot30sec/1snapshot.csv',delimiter = '\t', usecols=['VEHICLE','TIME', 'EASTING', 'NORTHING'])
# allowedMatsimOutput = matsimOutput
# for i in range(len(forbiddenIDs)):
#     allowedMatsimOutput = allowedMatsimOutput[(allowedMatsimOutput.VEHICLE !=int(forbiddenIDs[i]) )]
# # vehicleIDs = allowedMatsimOutput.VEHICLE.unique()
# # len(allowedMatsimOutput)
# allowedMatsimOutput.to_csv('D:/ax/gis/locationMappingToMezuroZones/gisInput.csv', header=True,index=False)
# todo: go to Arcmap and using the intersect geospatial tool make a file with Mezuro zone id

# snapData = pd.read_csv("D:/ax/gis/locationMappingToMezuroZones/gisOutputForPLU2.csv", usecols=['VEHICLE','TIME', 'EASTING', 'NORTHING', 'mzr_id'])
snapData = pd.read_csv('D:/ax/gis/locationMappingToMezuroZones/finalInputPython/finalInputPython.CSV', usecols=['VEHICLE','TIME', 'EASTING', 'NORTHING','mzr_id'])
# len(snapData)
snapData = snapData.sort_values(by=["VEHICLE","TIME"], ignore_index=True)
vehicleIDs = snapData.VEHICLE.unique()
snapshotInterval = 30 # todo: please enter snapshot interval from the simulation in seconds.
extendedData = pd.DataFrame()
sample3 = snapData[snapData.VEHICLE==34]
sample3['start'] = pd.to_timedelta(sample3['TIME'], unit = 's')
driver = vehicleIDs[0]
for driver in vehicleIDs: # fixme
    temp = snapData[snapData.VEHICLE == driver]
    # print(temp.VEHICLE.iloc[0])
    #__________generating last records of user _____________
    rep0 = temp.iloc[[-1], :]
    finishSimTime = temp.TIME.iloc[0]+24*3600 #todo: check to make sure temp.TIME.iloc[0] is int NOT 'str'
    numRep = floor((finishSimTime - temp.TIME.iloc[-1]) / snapshotInterval)
    rep0 = rep0.iloc[np.tile(np.arange(1), numRep)]
    rep0.index = range(numRep)
    for t in range(numRep):
        rep0.TIME.iloc[t] = temp.TIME.iloc[-1] + (t + 1) * snapshotInterval
    rep = rep0
    #__________generating in between records of user __________
    for k in range(len(temp)-1):
        if (temp.TIME.iloc[k+1]-temp.TIME.iloc[k]) > snapshotInterval:
            rep1 = temp.iloc[[k], :]
            numRep = floor((temp.TIME.iloc[k+1]-temp.TIME.iloc[k]) / snapshotInterval)
            rep1 = rep1.iloc[np.tile(np.arange(1), numRep)]
            rep1.index = range(numRep)
            for t in range(numRep):
                rep1.TIME.iloc[t] = temp.TIME.iloc[k] + (t + 1) * snapshotInterval
            rep = pd.concat([rep,rep1])
    extendedData = pd.concat([extendedData,rep])


print(time.time() - startTime) # it took 51670.19287753105 seconds (more than 14 hours)!!!!!


snapDataNew = pd.concat([snapData, extendedData])
snapDataNewSorted = snapDataNew.sort_values(by=["VEHICLE","TIME"])
snapDataSplit = np.array_split(snapDataNewSorted, 10)
for l in range(10):
    pd.DataFrame(snapDataSplit[l]).to_csv("D:/ax/gis/completePLUdata_30sec/completePLUdata_30sec_{0}.csv".format(l),
                                          header=True, index=False)
# pd.core.groupby.GroupBy.get_group(snapDataNewSorted, 'VEHICLE')
snapDataNewSorted.to_csv("D:/ax/gis/completePLUdata_30sec/completePLUdata_30sec.csv",
                                          header=True, index=False)
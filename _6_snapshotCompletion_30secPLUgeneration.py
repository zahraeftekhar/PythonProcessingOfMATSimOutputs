import pandas as pd
from shapely.geometry import Point
import time
from math import floor
import numpy as np
startTime = time.time()
# todo: go to Arcmap and using the intersect geospatial tool make a file with Mezuro zone id
# # # # # ____________________________________completing GIS output for out of Mezuro map points ______________________________
# precompletion = pd.read_csv('D:/ax/gis/MATSimOutput_snapshot30sec/GISoutput_PreCompletion.CSV',usecols=['mzr_id', 'VEHICLE','TIME','EASTING','NORTHING'])
# precompletion = precompletion.sort_values(by=["VEHICLE","TIME"])
# precompletion = precompletion.reset_index(drop=True)
# MATSimOutput = pd.read_csv('D:/ax/gis/MATSimOutput_snapshot30sec/MATSimCompleteData.CSV',usecols=['VEHICLE','TIME','EASTING','NORTHING'])
# MATSimOutput['VEHICLE'] = MATSimOutput.VEHICLE.astype(int)
# MATSimOutput['TIME'] = MATSimOutput.TIME.astype(int)
# MATSimOutput = MATSimOutput.sort_values(by=["VEHICLE","TIME"])
# MATSimOutput=MATSimOutput.reset_index(drop=True)
# completeData = pd.merge(precompletion, MATSimOutput, how='right', on=['VEHICLE','TIME'])
# (completeData.loc[:,'mzr_id'][completeData['mzr_id'].isna()]) = 0.0
# completeData = completeData.loc[:,['VEHICLE','TIME','EASTING_y','NORTHING_y','mzr_id']]
# completeData.columns = ['VEHICLE', 'TIME', 'EASTING', 'NORTHING', 'mzr_id']
# completeData = completeData.sort_values(by = ['VEHICLE', 'TIME'])
# completeData.to_csv('D:/ax/gis/locationMappingToMezuroZones/finalInputPython/finalInputPython.CSV', header=True,index=False)

# _________________________________________________________________________________________________
snapData = pd.read_csv('D:/ax/gis/locationMappingToMezuroZones/finalInputPython/finalInputPython.CSV', usecols=['VEHICLE','TIME', 'EASTING', 'NORTHING','mzr_id'])
snapData = snapData.sort_values(by=["VEHICLE","TIME"], ignore_index=True)
vehicleIDs = snapData.VEHICLE.unique()
snapshotInterval = 30 # todo: please enter snapshot interval from the simulation in seconds.
startTime = time.time()
concatData = pd.DataFrame()
# mm=0
for mm in range((len(vehicleIDs)//1000)): #176min for all users
    extendedData = pd.DataFrame()
    # if mm==(len(vehicleIDs)//500)-1:
    #     print((len(vehicleIDs)//500)-1)
    for i,driver in enumerate(vehicleIDs[1000*mm:1000*(mm+1)]): # fixme
        if i%100==0:
            print('{percentage} percent____{duration} sec'.format(percentage=i / len(vehicleIDs),
                                                                  duration=time.time() - startTime))
            # if (len(vehicleIDs)-100*hb)<100:
            #     hb=len(vehicleIDs)
        if driver in snapData.VEHICLE.values:
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
            del rep0# fixme :just added
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
    concatData=pd.concat([concatData,extendedData])
extendedData = pd.DataFrame()
for i,driver in enumerate(vehicleIDs[1000*(mm+1):len(vehicleIDs)]): # fixme
    if i%100==0:
        print('{percentage} percent____{duration} sec'.format(percentage=i / len(vehicleIDs),
                                                              duration=time.time() - startTime))
    if driver in snapData.VEHICLE.values:
        temp = snapData[snapData.VEHICLE == driver]
        #__________generating last records of user _____________
        rep0 = temp.iloc[[-1], :]
        finishSimTime = temp.TIME.iloc[0]+24*3600 #todo: check to make sure temp.TIME.iloc[0] is int NOT 'str'
        numRep = floor((finishSimTime - temp.TIME.iloc[-1]) / snapshotInterval)
        rep0 = rep0.iloc[np.tile(np.arange(1), numRep)]
        rep0.index = range(numRep)
        for t in range(numRep):
            rep0.TIME.iloc[t] = temp.TIME.iloc[-1] + (t + 1) * snapshotInterval
        rep = rep0
        del rep0# fixme :just added
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
concatData=pd.concat([concatData,extendedData])
print(time.time() - startTime) # it took 51670.19287753105 seconds (more than 14 hours)!!!!!


snapDataNew = pd.concat([snapData, concatData])
snapDataNewSorted = snapDataNew.sort_values(by=["VEHICLE","TIME"])
snapDataSplit = np.array_split(snapDataNewSorted, 10)
for l in range(10):
    pd.DataFrame(snapDataSplit[l]).to_csv("D:/ax/gis/completePLUdata_30sec/completePLUdata_30sec_{0}.csv".format(l),
                                          header=True, index=False)
# pd.core.groupby.GroupBy.get_group(snapDataNewSorted, 'VEHICLE')
snapDataNewSorted.to_csv("D:/ax/gis/completePLUdata_30sec/completePLUdata_30sec.csv",
                                          header=True, index=False)
import pandas as pd
from shapely.geometry import Point
import time
from math import floor
import numpy as np
startTime = time.time()
snapData = pd.read_csv("D:/ax/gis/observedLocations.csv", usecols=['VEHICLE','TIME', 'EASTING', 'NORTHING', 'mzr_id'])
snapData = snapData.sort_values(by=["VEHICLE","TIME"])
vehicleIDs = snapData.VEHICLE.unique()
#_______________________________________________________________________________________________________________________
# filling in the gaps due to activity durations to represent a fully PLU data: run time=19798.47118115425 seconds
#_______________________________________________________________________________________________________________________
beginSimTime = 10800 # todo: please emter the begining time of simulation in seconds, e.g., 3 am is 10800 seconds.
finishSimTime = 86400 # todo: please emter the finishing time of simulation in seconds, e.g., 12 am is 86400 seconds.
snapshotInterval = 30 # todo: please enter snapshot interval from the simulation in seconds.
extendedData = pd.DataFrame()
driver = vehicleIDs[0]
for driver in vehicleIDs: # fixme
    temp = snapData[snapData.VEHICLE == driver]
    baseTime = temp.TIME.iloc[0]
    if (baseTime-beginSimTime) > snapshotInterval:
        rep = temp.iloc[[0],:]
        numRep = floor((temp.TIME.iloc[0]-beginSimTime)/snapshotInterval)
        rep = rep.iloc[np.tile(np.arange(1),numRep)]
        rep.index = range(numRep)
        for t in range(numRep):
            rep.TIME.iloc[t] = baseTime - (t+1)*snapshotInterval
    if (finishSimTime - temp.TIME.iloc[-1]) > snapshotInterval:
        rep0 = temp.iloc[[-1],:]
        numRep = floor((finishSimTime - temp.TIME.iloc[-1])/snapshotInterval)
        rep0 = rep0.iloc[np.tile(np.arange(1),numRep)]
        rep0.index = range(numRep)
        for t in range(numRep):
            rep0.TIME.iloc[t] = temp.TIME.iloc[-1] + (t+1)*snapshotInterval
        rep = pd.concat([rep, rep0])
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
print(time.time() - startTime)
snapDataNew = pd.concat([snapData, extendedData])
snapDataNewSorted = snapDataNew.sort_values(by=["VEHICLE","TIME"])
# split_index0 =  0
# for l in range(1,floor(len(vehicleIDs)/10)):
#     splitID= l*floor(len(vehicleIDs)/10)
#     split_index = split_index0 + l*floor(len(vehicleIDs)/10)
snapDataSplit = np.array_split(snapDataNewSorted, 10)
for l in range(10):
    pd.DataFrame(snapDataSplit[l]).to_csv("D:/ax/gis/completePLUdata_30sec/completePLUdata_30sec_{0}.csv".format(l), header=True, index=False)
pd.core.groupby.GroupBy.get_group(snapDataNewSorted, 'VEHICLE')
# snapDataNew.to_csv("D:/ax/gis/completePLUdata_30sec_{0}.csv".format(l), header=True, index=False)

########################################################################################################################

# vehicleIDs = snapData.index.unique()
class SnapshotVehicles:
    def __init__(self, *args): pass
    def setTimeLocation(self,ID, times: list, Xs: list, Ys: list, mzr_ids: list ): #arguments as indicated should be a list
        self.id = ID
        self.location = []
        self.time = []
        for i in range(len(times)):
            tempPoint = LongLat()
            tempPoint.set_location(Xs[i],Ys[i])
            tempPoint.TAZ = mzr_ids[i]
            self.location = self.location + [tempPoint]
            self.time = self.time + [times[i]]


class LongLat:
    def __init__(self, *args):
        self.TAZ = 0
    def set_location(self, x, y):
        from shapely.geometry import Point
        self.location = Point(x, y)

    def changeCoordSys(self, initial: str = 'epsg:23031', final: str = 'epsg:28992'):
        from pyproj import Proj, transform
        from shapely.geometry import Point
        self.location = Point(transform(Proj(init=initial), Proj(init=final), self.location.x, self.location.y))

    def zoneMapping(self, onepolygon, polygonName):
        if (onepolygon.contains(self.location)):
            self.TAZ = polygonName


class TAZmap:
    def __init__(self): pass

    def set_map(self, value):
        import geopandas as gpd
        self.map = gpd.read_file(value)

map_mzr = TAZmap()
map_mzr.set_map('C:/Users/zahraeftekhar/PycharmProjects/XMLparsing1/netherlandsSHP/mezuro_areas_2018.shp')
inputs = map_mzr.map.geometry
tazNames = map_mzr.map.iloc[:].mzr_id
zoneZero = pd.Series(0)
matrixRowColNames = tuple(zoneZero.append(tazNames))
odsize = map_mzr.map.__len__() + 1
del map_mzr
startTime_OD = "06:00:00"
endTime_OD = "09:00:00"

# veh = SnapshotVehicles()
# veh.setTimeLocation(ID=vehicleIDs[0], times=snapData.iloc[snapData.index == vehicleIDs[0]].TIME.to_list(),
#                          Xs=snapData.iloc[snapData.index == vehicleIDs[0]].EASTING.to_list(),
#                          Ys=snapData.iloc[snapData.index == vehicleIDs[0]].NORTHING.to_list(),
#                     mzr_ids=snapData.iloc[snapData.index == vehicleIDs[0]].mzr_id.to_list())
# startT = time.time()
# for vehicleID in vehicleIDs[0:1000]:
#     veh = SnapshotVehicles()
#     veh.setTimeLocation(ID=vehicleID, times=snapData.iloc[snapData.index == vehicleID].TIME.to_list(),
#                          Xs=snapData.iloc[snapData.index == vehicleID].EASTING.to_list(),
#                          Ys=snapData.iloc[snapData.index == vehicleID].NORTHING.to_list(),
#                         mzr_ids=snapData.iloc[snapData.index == vehicleID].mzr_id.to_list())
# activityTime distribution in the other file
#
# print(time.time() - startT)
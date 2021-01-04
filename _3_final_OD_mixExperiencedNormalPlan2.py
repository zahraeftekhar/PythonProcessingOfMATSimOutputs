
################################################### CLASS ##############################################################
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


######################################### geo-spacial related packages #################################################
import pandas as pd
import time
# building the zero OD Matrix based on the number of zones in the SHP
map_mzr = TAZmap()
# map_mzr.set_map('D:/ax/gis/locationMappingToMezuroZones/amsterdamMezuroZones.shp')
map_mzr.set_map('/data/zahraeftekhar/research_temporal/input_base/amsterdamMezuroZones.shp')
inputs = map_mzr.map.geometry
# tazNames = map_mzr.map.iloc[:].mzr_id
# amsterdamMezuroZones = pd.read_csv('D:/ax/gis/locationMappingToMezuroZones/amsterdamMezuroZones.CSV', usecols=['mzr_id'])
amsterdamMezuroZones = pd.read_csv('/data/zahraeftekhar/research_temporal/input_base/amsterdamMezuroZones.CSV', usecols=['mzr_id'])
tazNames = amsterdamMezuroZones['mzr_id']
zoneZero = pd.Series(0)
matrixRowColNames = tuple(zoneZero.append(tazNames))
odsize=len(matrixRowColNames)
del map_mzr
ODstart = "15:30:00"
ODend = "18:30:00"
startTime_OD = pd.to_timedelta(ODstart)
endTime_OD = pd.to_timedelta(ODend)
######################################### importing XML file plan ######################################################
# from lxml import etree
import xml.etree.ElementTree as etree
# parser = etree.XMLParser(ns_clean=True, collect_ids=False)
# trueLocations = pd.read_csv("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples"
#                                "/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/ITERS/it.1/1.trueLocExperienced.csv")
trueLocations = pd.read_csv("/data/zahraeftekhar/research_temporal/output_base/1.trueLocExperienced.csv")
# itemlistExperienced= etree.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim_directory/matsim-example-project/scenarios/" \
#               "example_zahra/Amsterdam/original files/PlanWithOnlyCar_again_NoGeneric_NoZeroDuationActivity.xml").getroot().findall('person')
itemlistExperienced= etree.parse("/data/zahraeftekhar/research_temporal/output_base/PlanWithOnlyCar_again_NoGeneric_NoZeroDuationActivity.xml").getroot().findall('person')
######################################### deriving OD matrix from plan files ###########################################
import numpy as np
import pandas as pd
ODMatrix_df = pd.DataFrame(np.zeros((odsize, odsize), dtype=np.int32), columns=matrixRowColNames,
                           index=matrixRowColNames)  # creating empty OD matrix
# ODMatrix_home = pd.DataFrame(np.zeros((odsize, odsize), dtype=np.int32),
#                              columns=matrixRowColNames,
#                              index=matrixRowColNames)
# ODMatrix_work = pd.DataFrame(np.zeros((odsize, odsize), dtype=np.int32),
#                              columns=matrixRowColNames,
#                              index=matrixRowColNames)
# ODMatrix_other = pd.DataFrame(np.zeros((odsize, odsize), dtype=np.int32),
#                               columns=matrixRowColNames,
#                               index=matrixRowColNames)
person = itemlistExperienced[0]
m=0
mmm=1
start_time = time.time()
for m, person in enumerate(itemlistExperienced):
    if m == mmm * 1000:
        print('{percentage}%____{duration} min'.format(percentage=round((m / len(itemlistExperienced)*100)),
                                                              duration=round((time.time() - start_time)/60)))
        mmm += 1
    activityListPlan = itemlistExperienced[m].findall('plan/activity')
    activityListExperienced = itemlistExperienced[m].findall('plan/activity')
    truelocs = trueLocations[trueLocations['VEHICLE']==int(itemlistExperienced[m].get('id'))]
    truelocs = truelocs.reset_index(drop=True)
    j=1
    while j < len(activityListPlan):
        if j==len(activityListPlan)-1:
            start_time1 = pd.to_timedelta(activityListExperienced[j].get('start_time'))
            #____________________PLEASE ENTER THE BEGINING TIME OF THE SIMULATION _____________________
            end_time1 = pd.to_timedelta(activityListExperienced[0].get('end_time'))+ pd.to_timedelta('24:00:00')
            start_time2 = pd.to_timedelta(activityListExperienced[1].get('start_time'))+ pd.to_timedelta('24:00:00')
            endActivity = end_time1 # when using inconsistent timings : endActivity = min(end_time1, start_time2)
            startNewActivity = start_time2 # when using inconsistent timings : startNewActivity = max(end_time1, start_time2)
            if pd.to_timedelta('23:59:59')>=pd.to_timedelta(startTime_OD)>=pd.to_timedelta(start_time1):
                startTime_OD =ODstart
            else:
                startTime_OD = ODstart + pd.to_timedelta('24:00:00')
            if pd.to_timedelta('23:59:59')>=pd.to_timedelta(endTime_OD)>=pd.to_timedelta(start_time1):
                endTime_OD =pd.to_timedelta(ODend)
            else:
                endTime_OD = pd.to_timedelta(ODend)+ pd.to_timedelta('24:00:00')
        else:
            start_time1 = pd.to_timedelta(activityListExperienced[j].get('start_time'))
            # if j.__eq__(0):
            #     start_time1 = '03:00:00' #____________________PLEASE ENTER THE BEGINING TIME OF THE SIMULATION _____________________
            end_time1 = pd.to_timedelta(activityListExperienced[j].get('end_time'))
            start_time2 = pd.to_timedelta(activityListExperienced[j + 1].get('start_time'))
            endActivity = end_time1 # when using inconsistent timings : endActivity = min(end_time1, start_time2)
            startNewActivity = start_time2 # when using inconsistent timings : startNewActivity = max(end_time1, start_time2)
        if pd.to_timedelta(start_time1) <= pd.to_timedelta(startTime_OD) < pd.to_timedelta(startNewActivity):
            if endTime_OD <= endActivity:
                break
            else:
                while pd.to_timedelta(endTime_OD) > pd.to_timedelta(endActivity):
                    point1 = LongLat()
                    point1.set_location(x=float(truelocs.loc[j-1,'x']),
                                        y=float(truelocs.loc[j-1,'y'])) #point1.set_location(x=float(activityListPlan[j].get('x')), y=float(activityListPlan[j].get('y')))
                    point1.changeCoordSys()
                    for k in range(len(tazNames)):
                        point1.zoneMapping(inputs[k], tazNames[k])
                    origin = point1.TAZ
                    point2 = LongLat()
                    if j == len(activityListPlan) - 1:
                        point2.set_location(x=float(truelocs.loc[0,'x']),
                                            y=float(truelocs.loc[0,'y'])) #point2.set_location(x=float(activityListPlan[1].get('x')),y=float(activityListPlan[1].get('y')))
                        tempAcType = truelocs.loc[(0), 'activityType']
                    else:
                        point2.set_location(x=float(truelocs.loc[j,'x']),
                                        y=float(truelocs.loc[j,'y']))
                        tempAcType = truelocs.loc[(j), 'activityType']
                    point2.changeCoordSys()
                    for k in range(len(tazNames)):
                        point2.zoneMapping(inputs[k], tazNames[k])
                    destination = point2.TAZ
                    # if tempAcType == 'home':
                    #     ODMatrix_home[origin][destination] = ODMatrix_home[origin][destination] + 1
                    # elif tempAcType == 'work':
                    #     ODMatrix_work[origin][destination] = ODMatrix_work[origin][destination] + 1
                    # else:
                    #     ODMatrix_other[origin][destination] = ODMatrix_other[origin][destination] + 1
                    ODMatrix_df[origin][destination] = ODMatrix_df[origin][destination] + 1
                    j += 1
                    if j > len(activityListPlan) - 1: break
                    if j== len(activityListPlan)-1:
                        end_time1 = pd.to_timedelta(activityListExperienced[0].get('end_time'))+pd.to_timedelta('24:00:00')
                    else:
                        end_time1 = pd.to_timedelta(activityListExperienced[j].get('end_time'))
                    endActivity = end_time1 # when using inconsistent timings :endActivity = min(end_time1, start_time2)
                break
            # continue

        j += 1
    # print(m/len(itemlistPlan), '______',(time.time() - start_time))
    # print(np.sum(np.sum(ODMatrix_df, axis=0))) #13353 in the last run
# print(np.sum(np.sum(ODMatrix_df, axis=0)))
print(time.time() - start_time)
# TXTFileName = "D:/ax/OD({start1}-{start2}_{end1}-{end2}).CSV".format(start1 = ODstart[0:2],start2 = ODstart[3:5],
#                                                                      end1 = ODend[0:2], end2 = ODend[3:5])
TXTFileName = "/data/zahraeftekhar/research_temporal/output_base/OD({start1}-{start2}_{end1}-{end2}).CSV".format(start1 = ODstart[0:2],start2 = ODstart[3:5],
                                                                     end1 = ODend[0:2], end2 = ODend[3:5])
# ODMatrix_home.to_csv('D:/ax/OD({a}-{b}_{c}-{d})_home.CSV'.
#                      format( a=ODstart[0:2], b=ODstart[3:5], c=ODend[0:2], d=ODend[3:5]),
#                      header=True, index=True)
# ODMatrix_work.to_csv('D:/ax/OD({a}-{b}_{c}-{d})_work.CSV'.
#                      format( a=ODstart[0:2], b=ODstart[3:5], c=ODend[0:2], d=ODend[3:5]),
#                      header=True, index=True)
# ODMatrix_other.to_csv('D:/ax/OD({a}-{b}_{c}-{d})_other.CSV'.
#                       format( a=ODstart[0:2], b=ODstart[3:5], c=ODend[0:2], d=ODend[3:5]),
#                       header=True, index=True)
ODMatrix_df.to_csv(TXTFileName, index=True, header=True)
################ FINAL decision: this code fraction works for OD estimation from ground truth .XML data

################ Elapsed time: 3286 seconds,sum OD:13366
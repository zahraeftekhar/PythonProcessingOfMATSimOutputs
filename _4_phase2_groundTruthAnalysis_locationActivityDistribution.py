import numpy as np
import pandas as pd
import time
import xmltodict
from shapely.geometry import Point
######################################### importing XML file plan ######################################################
itemlistExperienced= open("D:/ax/gis/phase2/PlanWithOnlyCar_again_NoGeneric_NoZeroDuationActivity.xml","rb")
# itemlistExperienced= open("/data/zahraeftekhar/research_temporal/phase2/PlanWithOnlyCar_again_NoGeneric_NoZeroDuationActivity.xml","rb")
itemlistExperienced = xmltodict.parse(itemlistExperienced)
itemlistExperienced = pd.DataFrame.from_dict(itemlistExperienced['population'])

trueLocations = pd.read_csv("D:/ax/gis/phase2/1.trueLocExperienced.csv")

# trueLocations = pd.read_csv("/data/zahraeftekhar/research_temporal/phase2/1.trueLocExperienced.csv")

plans = {}
for person in itemlistExperienced.iloc[:,0]:
    try:
        ends = [pd.to_timedelta(d['@end_time']) for d in person['plan']['activity'] if '@end_time' in d]
        starts = [pd.to_timedelta(d['@start_time']) for d in person['plan']['activity'] if '@start_time' in d]
        # durations = [(ends[d+1]-starts[d]) for d in range(len(ends)) if (d != len(starts)-1) else (ends[0]-starts[-1]) ]
        durations = [(ends[d+1]-starts[d]) if (d != len(starts)-1) else ((ends[0]+pd.to_timedelta(1, unit='d'))-starts[-1]) for d in range(len(ends))]
        types = [d['@type'] for d in person['plan']['activity'] if '@type' in d][1:]
        types = [d if ((d=="home") or (d== "work")) else ("other") for d in types]
        x = trueLocations[trueLocations.loc[:,"VEHICLE"]==int((person)['@id'])].loc[:,"x"]
        y = trueLocations[trueLocations.loc[:,"VEHICLE"]==int((person)['@id'])].loc[:,"y"]
        plans[(person)['@id']] = {'start': np.array(starts),
                                                       'duration': np.array(durations), 'type': np.array(types),
                                  "x":np.array(x), "y":np.array(y)}
    except TypeError :
        print((person)['@id'])
    except IndexError:
        ends = ends[0:-1]
        durations = [(ends[d + 1] - starts[d]) if (d != len(starts) - 1) else (
                    (ends[0] + pd.to_timedelta(1, unit='d')) - starts[-1]) for d in range(len(ends))]
        types = [d['@type'] for d in person['plan']['activity'] if '@type' in d][1:]
        types = [d if ((d == "home") or (d == "work")) else ("other") for d in types]
        x = trueLocations[trueLocations.loc[:, "VEHICLE"] == int((person)['@id'])].loc[:, "x"]
        y = trueLocations[trueLocations.loc[:, "VEHICLE"] == int((person)['@id'])].loc[:, "y"]
        # locs = trueLocations[trueLocations.loc[:, "VEHICLE"] == int((person)['@id'])].loc[:, ["x", "y"]]
        plans[(person)['@id']] = {'start': np.array(starts),
                                  'duration': np.array(durations), 'type': np.array(types),
                                  "x": np.array(x) , "y":np.array(y)}
import pickle
with open('D:/ax/gis/phase2/activities.pickle', 'wb') as handle:
    pickle.dump(plans, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('/data/zahraeftekhar/research_temporal/phase2/activities.pickle', 'wb') as handle:
#     pickle.dump(plans, handle, protocol=pickle.HIGHEST_PROTOCOL)

durations= np.array([])
starts = np.array([])
types = np.array([])
xs = np.array([])
ys = np.array([])
ids = np.array([])
for ID, val in plans.items() :
    durations = np.hstack((durations,val['duration']))
    starts = np.hstack((starts,val['start']))
    types = np.hstack((types,val['type']))
    xs = np.hstack((xs,val['x']))
    ys = np.hstack((ys,val['y']))
    ids = np.hstack((ids,np.repeat(ID, len(val['duration']))))
activities = pd.DataFrame()
activities["start"] = starts
activities["duration"] = durations
activities["type"] = types
activities['id'] = ids
activities["x"] = xs
activities["y"] = ys
activities.to_csv("D:/ax/gis/phase2/activities.CSV",header=True,index=False)
# activities.to_csv("/data/zahraeftekhar/research_temporal/phase2/activities.CSV",header=True,index=False)
with open('D:/ax/gis/phase2/activities_df.pickle', 'wb') as handle:
    pickle.dump(activities, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('/data/zahraeftekhar/research_temporal/phase2/activities_df.pickle', 'wb') as handle:
#     pickle.dump(activities, handle, protocol=pickle.HIGHEST_PROTOCOL)
plans = {}
for person in itemlistExperienced.iloc[:,0]:
    try:
        starts = [pd.to_timedelta(d['@dep_time']) for d in person['plan']['leg'] if '@dep_time' in d]
        durations = [pd.to_timedelta(d['@trav_time']) for d in person['plan']['leg'] if '@trav_time' in d]
        plans[(person)['@id']] = {'start': np.array(starts),'duration': np.array(durations)}
    except KeyError :
        print((person)['@id'])

with open('D:/ax/gis/phase2/trips.pickle', 'wb') as handle:
    pickle.dump(plans, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('/data/zahraeftekhar/research_temporal/phase2/trips.pickle', 'wb') as handle:
#     pickle.dump(plans, handle, protocol=pickle.HIGHEST_PROTOCOL)

durations_trip= np.array([])
starts_trip = np.array([])
ids_trip = np.array([])
for ID, val in plans.items() :
    durations_trip = np.hstack((durations_trip,val['duration']))
    starts_trip = np.hstack((starts_trip,val['start']))
    ids_trip = np.hstack((ids_trip,np.repeat(ID, len(val['duration']))))
trips = pd.DataFrame()
trips['id'] = ids_trip
trips["start"] = starts_trip
trips["duration"] = durations_trip
trips.to_csv("D:/ax/gis/phase2/trips.CSV",header=True,index=False)
# trips.to_csv("/data/zahraeftekhar/research_temporal/phase2/trips.CSV",header=True,index=False)
with open('D:/ax/gis/phase2/trips_df.pickle', 'wb') as handle:
    pickle.dump(trips, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('/data/zahraeftekhar/research_temporal/phase2/trips_df.pickle', 'wb') as handle:
#     pickle.dump(trips, handle, protocol=pickle.HIGHEST_PROTOCOL)
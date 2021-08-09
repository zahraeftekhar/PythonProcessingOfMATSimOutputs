import pandas as pd
from scipy.stats import gaussian_kde
import time
import numpy as np
import pickle

# #____________________________________ clustering records ________________________________________
# files = [30,60,300,600,900,1200,1500,1800,2100,2400,2700,3000,3300,3600,4500,5400,6300,7200]
#
# # interval = 30 #todo: please enter the interval between snapData records
# startTime = time.time()
# for q in files: #27 min for each file
#     print("start of interval {q} ______ time:{t}min ".format(q=q,t =(time.time()-startTime)//60 ))
#     clusterData = {}
#     snapDataSplit = pd.read_csv("Y:/ZahraEftekhar/_1_/finalRun_noZeroDuration2/completePLUdata_{number}sec"
#                                 "/completePLUdata_{number}sec.csv".format(number=q)) #38.5 sec to load 4G data
#     snapDataSplit[['mzr_id', 'VEHICLE']] = snapDataSplit[['mzr_id', 'VEHICLE']].astype(int)
#     snapDataSplit = snapDataSplit.sort_values(by=["VEHICLE","TIME"])
#     vehicleIDs = snapDataSplit.VEHICLE.unique()
#     vehicleIDs = np.sort(vehicleIDs)
#
#     startTime = time.time()
#     for i,person in snapDataSplit.groupby(['VEHICLE']):
#         if i%100==0:
#             print('{percentage} percent____{duration} sec'.format(percentage = i/len(vehicleIDs), duration = time.time()-startTime))
#         person = person.reset_index(drop=True)
#         newD = pd.DataFrame()
#         for k,v in person.groupby((person['mzr_id'].shift() != person['mzr_id']).cumsum()):
#             newD = newD.append(v.iloc[0,:])
#         d = np.array(newD["TIME"].shift(periods=-1, fill_value=v.iloc[-1,:]["TIME"])) - np.array(
#             newD["TIME"])
#         clusterData[person.loc[0,"VEHICLE"]] = {"mzr_id":np.array(newD["mzr_id"]),'start':np.array(
#             newD["TIME"]),"x":np.array(newD["EASTING"]),"y":np.array(newD["NORTHING"]),"id":np.array(
#                 newD["VEHICLE"]),"duration": np.array(d) }
#     with open('D:/ax/gis/phase2/clusterData_{number}sec.pickle'.format(number = q), 'wb') as handle:
#         pickle.dump(clusterData, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # _____________________________________________________________________________________________________________________
#_______________ location type identification _____________________________________________________
files = [2400,2700,3000,3300,3600,4500,5400,6300,7200]
# locProb = pd.read_csv("D:/ax/gis/phase2/locProbability.csv",usecols=["mzr_id","home","work","other"])
locProb = pd.read_csv("/data/zahraeftekhar/research_temporal/phase2/locProbability.csv",usecols=["mzr_id","home","work","other"])
locProb["home"][locProb["home"]==0] = 1/3
locProb["work"][locProb["work"]==0] = 1/3
locProb["other"][locProb["other"]==0] = 1/3
locProb = locProb.append({"mzr_id":0,"home":1/3,"work":1/3,"other":1/3}, ignore_index=True)
startTime = time.time()
for q in files:
    startTime = time.time()
    print("start of interval {q} ______ time:{t}min ".format(q=q, t=(time.time() - startTime) // 60))
    seeds = range(101,126)
    for seed in seeds:
        print("******** seed is {inin} sec ******* time:{t}min".format(inin=seed,t=(time.time() - startTime) // 60))
        with open('/data/zahraeftekhar/research_temporal/phase2/trainingTrip_seed{s}.pickle'.format(s=seed), 'rb') as handle:
            trip = pickle.load(handle)
        with open('/data/zahraeftekhar/research_temporal/phase2/trainingActivity_seed{s}.pickle'.format(s=seed), 'rb') as handle:
            activity = pickle.load(handle)
        with open('/data/zahraeftekhar/research_temporal/phase2/trainingHome_seed{s}.pickle'.format(s=seed), 'rb') as handle:
            home = pickle.load(handle)
        with open('/data/zahraeftekhar/research_temporal/phase2/trainingWork_seed{s}.pickle'.format(s=seed), 'rb') as handle:
            work = pickle.load(handle)
        with open('/data/zahraeftekhar/research_temporal/phase2/trainingOther_seed{s}.pickle'.format(s=seed), 'rb') as handle:
            other = pickle.load(handle)
        prior_activity = len(activity)/(len(activity) + len(trip))
        prior_trip = len(trip)//(len(activity) + len(trip))
        prior_home = len(home)/(len(home) + len(work) + len(other))
        prior_work = len(work)/(len(home) + len(work) + len(other))
        prior_other = len(other)/(len(home) + len(work) + len(other))

        with open('/data/zahraeftekhar/research_temporal/phase2/clusterData_{number}sec.pickle'.format(number=q), 'rb') as handle:
            clusterData = pickle.load(handle)
        identified_clusterData = {}
        for i in clusterData.keys():
            d = clusterData[i]["duration"]
            st = clusterData[i]["start"]
            zone = clusterData[i]["mzr_id"]
            ps = prior_activity* gaussian_kde(np.array(activity["duration"].dt.total_seconds())).pdf(
                d)* gaussian_kde(np.array(activity["start"].dt.total_seconds())).pdf(st)
            pp = prior_activity* gaussian_kde(np.array(trip["duration"].dt.total_seconds())).pdf(
                d)* gaussian_kde(np.array(trip["start"].dt.total_seconds())).pdf(st)
            logprob_stay = ps/(ps+pp)
            clusterData[i]["prob_stay"] =logprob_stay
            identified_clusterData[i] = {"mzr_id":np.array(zone[np.where(logprob_stay>=0.5)]),
                                         'start':np.array(st[np.where(logprob_stay>=0.5)]),
                                         'duration': np.array(d[np.where(logprob_stay >= 0.5)]),
                                         "x":np.array(clusterData[i]["x"][np.where(logprob_stay>=0.5)]),
                                         "y":np.array(clusterData[i]["y"][np.where(logprob_stay>=0.5)]),
                                         "id":np.array(clusterData[i]["id"][np.where(logprob_stay>=0.5)]),
                                         "stayProb":logprob_stay[logprob_stay>=0.5],
                                         "loc_homeProb": np.array((pd.DataFrame(
                                             zone[np.where(logprob_stay>=0.5)],
                                             columns = ["mzr_id"]).merge(
                                             locProb,on = ["mzr_id"],how = "inner"))["home"]),
                                         "loc_workProb": np.array((pd.DataFrame(
                                             zone[np.where(logprob_stay>=0.5)],
                                             columns = ["mzr_id"]).merge(
                                             locProb,on = ["mzr_id"],how = "inner"))["work"]),
                                         "loc_otherProb": np.array((pd.DataFrame(
                                             zone[np.where(logprob_stay>=0.5)],
                                             columns = ["mzr_id"]).merge(
                                             locProb,on = ["mzr_id"],how = "inner"))["other"])}
            ph = prior_home * gaussian_kde(np.array(home["duration"].dt.total_seconds())).pdf(
                identified_clusterData[i]["duration"]) * gaussian_kde(np.array(home["start"].dt.total_seconds( \
                ))).pdf(
                identified_clusterData[i]["start"]) * identified_clusterData[i]["loc_homeProb"]
            pw = prior_work * gaussian_kde(np.array(work["duration"].dt.total_seconds())).pdf(
                identified_clusterData[i]["duration"]) * gaussian_kde(np.array(work["start"].dt.total_seconds( \
                ))).pdf(
                identified_clusterData[i]["start"]) * identified_clusterData[i]["loc_workProb"]
            po = prior_other * gaussian_kde(np.array(other["duration"].dt.total_seconds())).pdf(
                identified_clusterData[i]["duration"]) * gaussian_kde(np.array(other["start"].dt.total_seconds( \
                ))).pdf(
                identified_clusterData[i]["start"]) * identified_clusterData[i]["loc_otherProb"]
            identified_clusterData[i]["homeProb"] = ph/(ph+pw+po)
            identified_clusterData[i]["workProb"] = pw / (ph + pw + po)
            identified_clusterData[i]["otherProb"] = po / (ph + pw + po)
            identified_clusterData[i]["activity"] = np.argmax(np.vstack((identified_clusterData[i]["homeProb"],
                                                                         identified_clusterData[i]["workProb"],
                                                                         identified_clusterData[i]["otherProb"])),
                                                              axis=0)
        with open('/data/zahraeftekhar/research_temporal/phase2/clusterData_identified_{number}sec_seed{ss}.pickle'.format(number=q,ss=seed),
                  'wb') as handle:
            pickle.dump(identified_clusterData, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('/data/zahraeftekhar/research_temporal/phase2/clusterData_identifiedStayOnly_{number}sec_seed{ss}.pickle'.format(number=q,ss=seed),
                  'wb') as handle:
            pickle.dump(clusterData, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # _______________________reconsidering activity type based on home loc_________________

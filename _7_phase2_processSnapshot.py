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
locProb = pd.read_csv("D:/ax/gis/phase2/locProbability.csv",usecols=["mzr_id","home","work","other"])
locProb["home"][locProb["home"]==0] = 1/3
locProb["work"][locProb["work"]==0] = 1/3
locProb["other"][locProb["other"]==0] = 1/3
locProb = locProb.append({"mzr_id":0,"home":1/3,"work":1/3,"other":1/3}, ignore_index=True)
files = [7200]
startTime = time.time()
for q in files:
    startTime = time.time()
    print("start of interval {q} ______ time:{t}min ".format(q=q, t=(time.time() - startTime) // 60))
    seeds = range(101,126)
    for seed in seeds:
        print("******** seed is {inin} sec ******* time:{t}min".format(inin=seed,t=(time.time() - startTime) // 60))
        with open('D:/ax/gis/phase2/trainingTrip_seed{s}.pickle'.format(s=seed), 'rb') as handle:
            trip = pickle.load(handle)
        with open('D:/ax/gis/phase2/trainingActivity_seed{s}.pickle'.format(s=seed), 'rb') as handle:
            activity = pickle.load(handle)
        with open('D:/ax/gis/phase2/trainingHome_seed{s}.pickle'.format(s=seed), 'rb') as handle:
            home = pickle.load(handle)
        with open('D:/ax/gis/phase2/trainingWork_seed{s}.pickle'.format(s=seed), 'rb') as handle:
            work = pickle.load(handle)
        with open('D:/ax/gis/phase2/trainingOther_seed{s}.pickle'.format(s=seed), 'rb') as handle:
            other = pickle.load(handle)
        prior_activity = len(activity)/(len(activity) + len(trip))
        prior_trip = len(trip)//(len(activity) + len(trip))
        prior_home = len(home)/(len(home) + len(work) + len(other))
        prior_work = len(work)/(len(home) + len(work) + len(other))
        prior_other = len(other)/(len(home) + len(work) + len(other))

        with open('D:/ax/gis/phase2/clusterData_{number}sec.pickle'.format(number=q), 'rb') as handle:
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
        with open('D:/ax/gis/phase2/clusterData_identified_{number}sec_seed{ss}.pickle'.format(number=q,ss=seed),
                  'wb') as handle:
            pickle.dump(identified_clusterData, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('D:/ax/gis/phase2/clusterData_identifiedStayOnly_{number}sec_seed{ss}.pickle'.format(number=q,ss=seed),
                  'wb') as handle:
            pickle.dump(clusterData, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # _______________________reconsidering activity type based on home loc_________________
# seeds = range(101, 126)
# startTime = time.time()
# for seed in seeds:
#     print("************** seed is {inin} and time is {tt} min***************".format(inin=seed,tt=round((time.time() - startTime) / 60)))
#     anchorLocs = pd.DataFrame()
#     for t in range(0,10):
#         # anchor = pd.read_csv('D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/anchorLocsPLU_30sec_seed{ss}_{number}.CSV'.format(number = t))
#         anchor = pd.read_csv('/data/zahraeftekhar/research_temporal/completePLUdata_30sec/clusterPLU_30sec/anchorLocsPLU_30sec_seed{ss}_{number}.CSV'.format(ss=seed,number = t))
#         anchorLocs= anchorLocs.append(anchor,ignore_index=True)
#     vehicleIDs = anchorLocs['VEHICLE'].unique()
#     for i in range(len(vehicleIDs)): #range(len(vehicleIDs))
#             temp = (anchorLocs[anchorLocs.VEHICLE == vehicleIDs[i]])
#             if len(anchorLocs.loc[temp.index[temp['activity'] == 'home'],])>0:
#                 homeEASTING = anchorLocs.loc[temp.index[temp['activity'] == 'home'],'EASTING'].reset_index(drop=True)[len(anchorLocs.loc[temp.index[temp['activity'] == 'home'],'EASTING'])-1]
#                 homeNORTHING = anchorLocs.loc[temp.index[temp['activity'] == 'home'],'NORTHING'].reset_index(drop=True)[len(anchorLocs.loc[temp.index[temp['activity'] == 'home'],'NORTHING'])-1]
#                 for j in temp.index.drop(temp.index[temp['activity'] == 'home']):
#                     if ((float(anchorLocs.loc[j,'EASTING'])-float(homeEASTING))**2+ (float(anchorLocs.loc[j,'NORTHING'])-float(homeNORTHING))**2)**0.5<300:
#                         anchorLocs.loc[j,'activity'] = 'home'
#     # anchorLocs.to_csv("D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/anchorLocsPLU_30sec_seed{ss}.CSV".format(ss=seed),header=True,index=False)
#     anchorLocs.to_csv("/data/zahraeftekhar/research_temporal/completePLUdata_30sec/clusterPLU_30sec/anchorLocsPLU_30sec_seed{ss}.CSV".format(ss=seed),header=True,index=False)
# # __________ estimation OD matrix ___________  todo: put Amsterdam zoning map with data point on it in the report
# # anchorLocs = pd.DataFrame()
# # for t in range(0,10):
# #     anchor = pd.read_csv('D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/anchorLocsPLU_30sec_{number}.CSV'.format(number = t))
# #     anchorLocs= anchorLocs.append(anchor,ignore_index=True)
# seeds = range(101, 126)
# for seed in seeds:
#     # anchorLocs = pd.read_csv("D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/anchorLocsPLU_30sec_seed{ss}.CSV".format(ss=seed))
#     anchorLocs = pd.read_csv("/data/zahraeftekhar/research_temporal/completePLUdata_30sec/clusterPLU_30sec/anchorLocsPLU_30sec_seed{ss}.CSV".format(ss=seed))
#     anchorLocs['end(sec)'] = anchorLocs['start(sec)']+anchorLocs['duration(sec)']
#     # amsterdamMezuroZones = pd.read_csv('D:/ax/gis/locationMappingToMezuroZones/amsterdamMezuroZones.CSV', usecols=['mzr_id'])
#     amsterdamMezuroZones = pd.read_csv('/data/zahraeftekhar/research_temporal/input_base/amsterdamMezuroZones.CSV', usecols=['mzr_id'])
#     tazNames = amsterdamMezuroZones['mzr_id'] #5333 is also included but not in amsterdam so '0' zone represent it
#     zoneZero = pd.Series(0)
#     matrixRowColNames = tuple(zoneZero.append(tazNames))
#     odsize=len(matrixRowColNames)
#     ODstart = "06:30:00"
#     ODend = "09:30:00"
#     startTime_OD = pd.to_timedelta(ODstart)
#     endTime_OD = pd.to_timedelta(ODend)
#     ODMatrix_df = pd.DataFrame(np.zeros((odsize, odsize), dtype=np.int32),
#                                columns=matrixRowColNames,
#                                index=matrixRowColNames)  # creating empty OD matrix
#     vehicleIDs = anchorLocs.VEHICLE.unique().astype(int)
#     start_time = time.time()
#     m=0
#     for m in range(len(vehicleIDs)): #range(len(vehicleIDs))
#         activityList = anchorLocs[anchorLocs.VEHICLE == vehicleIDs[m]]
#
#         activityList = activityList.reset_index(drop=True)
#         j=1
#         while j < len(activityList):
#             if j==len(activityList)-1:
#                 start_time1 = pd.to_timedelta(activityList.loc[j,'start(sec)'], unit='sec')
#                 end_time1 = pd.to_timedelta(activityList.loc[j,'end(sec)'], unit='sec')
#                 start_time2 = pd.to_timedelta(activityList.loc[0,'start(sec)'], unit='sec')+ pd.to_timedelta('24:00:00')
#                 endActivity = end_time1
#                 startNewActivity = start_time2
#                 if pd.to_timedelta('23:59:59')>=pd.to_timedelta(startTime_OD)>=pd.to_timedelta(start_time1):
#                     startTime_OD =ODstart
#                 else:
#                     startTime_OD = ODstart + pd.to_timedelta('24:00:00')
#                 if pd.to_timedelta('23:59:59')>=pd.to_timedelta(endTime_OD)>=pd.to_timedelta(start_time1):
#                     endTime_OD =ODend
#                 else:
#                     endTime_OD = ODend+ pd.to_timedelta('24:00:00')
#             else:
#                 start_time1 = pd.to_timedelta(activityList.loc[j,'start(sec)'], unit='sec')
#                 end_time1 = pd.to_timedelta(activityList.loc[j,'end(sec)'], unit='sec')
#                 start_time2 = pd.to_timedelta(activityList.loc[(j + 1),'start(sec)'], unit='sec')
#                 endActivity = end_time1 # when using inconsistent timings : endActivity = min(end_time1, start_time2)
#                 startNewActivity = start_time2 # when using inconsistent timings : startNewActivity = max(end_time1, start_time2)
#             if pd.to_timedelta(start_time1) <= pd.to_timedelta(startTime_OD) < pd.to_timedelta(startNewActivity):
#                 if endTime_OD <= endActivity:
#                     break
#                 else:
#                     while endTime_OD > endActivity:
#                         origin = activityList.loc[j,'mzr_id']
#                         if j == len(activityList) - 1:
#                             destination = activityList.loc[(0),'mzr_id']
#                         else:
#                             destination = activityList.loc[(j+1), 'mzr_id']
#                         ODMatrix_df[origin][destination] = ODMatrix_df[origin][destination] + 1
#                         j += 1
#                         if j > len(activityList) - 1: break
#                         end_time1 = pd.to_timedelta(activityList.loc[j,'end(sec)'], unit='sec')
#                         endActivity = end_time1 # when using inconsistent timings :endActivity = min(end_time1, start_time2)
#                     break
#                 # continue
#
#             j += 1
#         # print(activityList.loc[0,'VEHICLE'], end='____')
#
#     print(np.sum(np.sum(ODMatrix_df, axis=0))) #13264 for 30sec interval ***********
#     ODMatrix_df.to_csv('/data/zahraeftekhar/research_temporal/completePLUdata_30sec/OD({a}-{b}_{c}-{d})_seed{ss}.CSV'.
#                        format(a=ODstart[0:2], b= ODstart[3:5], c = ODend[0:2], d=ODend[3:5],ss=seed),header=True, index=True)
#     # ODoriginal = pd.read_csv("D:/ax/OD(06-30_09-30).CSV")
#     ODoriginal = pd.read_csv("/data/zahraeftekhar/research_temporal/output_base/OD(06-30_09-30).CSV")
#     ODoriginal = ODoriginal.set_index('Unnamed: 0')
#     print(np.sum(np.sum(ODoriginal, axis=0))) #13353 ************
# # #
# #
# # # anchorrrrrLoc = anchorLocs
# # # anchorrrrrLoc['start'] = pd.to_timedelta(anchorLocs.loc[:,'start(sec)'],unit='s')
# # # anchorrrrrLoc['end'] = pd.to_timedelta(anchorLocs.loc[:,'end(sec)'],unit='s')
# # # anchorrrrrLoc['ID'] = anchorrrrrLoc['VEHICLE']
# # # anchorrrrrLoc['duration'] = pd.to_timedelta(anchorLocs.loc[:,'duration(sec)'],unit='s')
# # _________________________ number of activities per user ______________________________
# # anchorLocs = pd.DataFrame()
# # for t in range(0,10):
# #     anchor = pd.read_csv('D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/anchorLocsPLU_30sec_{number}.CSV'.format(number = t))
# #     anchorLocs= anchorLocs.append(anchor,ignore_index=True)
# # anchorLocs = pd.read_csv("D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/anchorLocsPLU_30sec.CSV")
# # anchorLocs['end(sec)'] = anchorLocs['start(sec)']+anchorLocs['duration(sec)']
# # from lxml import etree
# # parser = etree.XMLParser(ns_clean=True, collect_ids=False)
# # itemlistExperienced= etree.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples"
# #                                "/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.experienced_plans_Nogeneric(all allowed).xml").getroot()
# # vehicleIDs = anchorLocs['VEHICLE'].unique()
# # i=0
# # nActivity = pd.DataFrame(columns=['VEHICLE', 'GT', 'PLU'])
# # # mmm=1
# # for i in range(len(vehicleIDs)):
# #     # if i==mmm*100:
# #     #     print('{percentage} percent____{duration} sec'.format(percentage = i/len(vehicleIDs), duration = time.time()-startTime))
# #     #     mmm +=1
# #     nActivity.loc[i,'VEHICLE'] = vehicleIDs[i]
# #     nActivity.loc[i,'PLU'] = len(anchorLocs.loc[anchorLocs.loc[:,'VEHICLE']==vehicleIDs[i],"VEHICLE"])
# #     nActivity.loc[i,'GT'] = len(itemlistExperienced.find('person[@id="{id}"]'.format(id=vehicleIDs[i].astype(int))).findall('plan/activity'))-1
# # nActivity['comparison'] = nActivity['PLU']==nActivity['GT']
# # sum(nActivity['comparison'])/len(nActivity)
# # sum(nActivity['PLU'])
# # sum(nActivity['GT'])
# # correctNActivity = nActivity[(nActivity['comparison']==1)].VEHICLE
# # correctNActivity = correctNActivity.reset_index(drop=True)
# # # _________________________________ activity type_________________________
# # activityType = pd.DataFrame(columns=['VEHICLE', 'PLU','GT'])
# #
# # for i in range(len(correctNActivity)):
# #     activityType.loc[i,'VEHICLE'] = correctNActivity[i]
# #     testActivityPLU = []
# #     for j in (anchorLocs.loc[anchorLocs.loc[:,'VEHICLE']==correctNActivity[i],"activity"]):
# #         testActivityPLU+=[j]
# #     activityType.at[i,'PLU'] = testActivityPLU
# #     testActivityGT = []
# #     for j in (((itemlistExperienced.find('person[@id="{id}"]'.format(id=correctNActivity[i].astype(int))).findall('plan/activity')[1:]))):
# #         if j.get('type')!='home' and j.get('type')!='work':
# #             testActivityGT += ['other']
# #         else:
# #             testActivityGT += [j.get('type')]
# #     activityType.at[i, 'GT'] = testActivityGT
# #
# # activityType['comparison'] = activityType['PLU']==activityType['GT']
# # sum(activityType['comparison'])/len(activityType)
# # # ___________________________________________________________________
# # activityType = pd.DataFrame(columns=['VEHICLE', 'PLU','GT'])
# # testActivityGT = []
# # testActivityPLU = []
# # starts = []
# # durations = []
# # for i in range(len(correctNActivity)):
# #     # activityType.loc[i,'VEHICLE'] = correctNActivity[i]
# #     for j in (anchorLocs.loc[anchorLocs.loc[:,'VEHICLE']==correctNActivity[i],"activity"]):
# #         testActivityPLU+=[j]
# #     for l in range(len(anchorLocs.loc[anchorLocs.loc[:,'VEHICLE']==correctNActivity[i],"start(sec)"])):
# #             starts+=[(anchorLocs.loc[anchorLocs.loc[:,'VEHICLE']==correctNActivity[i],"start(sec)"].reset_index(drop=True))[l]]
# #             durations+=[anchorLocs.loc[anchorLocs.loc[:,'VEHICLE']==correctNActivity[i],"duration(sec)"].reset_index(drop=True)[l]]
# #
# #     # activityType.at[i,'PLU'] = testActivityPLU
# #     for k in (((itemlistExperienced.find('person[@id="{id}"]'.format(id=correctNActivity[i].astype(int))).findall('plan/activity')[1:]))):
# #         if k.get('type')!='home' and k.get('type')!='work':
# #             testActivityGT += ['other']
# #         else:
# #             testActivityGT += [k.get('type')]
# # activityType = pd.DataFrame(columns=['PLU','GT'])
# # activityType['PLU'] = testActivityPLU
# # activityType['GT'] = testActivityGT
# # activityType['start(sec)'] = starts
# # activityType['duration(sec)'] = durations
# # activityType['comparison'] = activityType['PLU']==activityType['GT']
# # # ________________________________________________________________________
# # sum(activityType['comparison'])/len(activityType)
# # wrongActivityPredictions = activityType[activityType.comparison==0]
# # homeWrongActivityPredictions = wrongActivityPredictions[wrongActivityPredictions.GT=='home']
# # workWrongActivityPredictions = wrongActivityPredictions[wrongActivityPredictions.GT=='work']
# # otherWrongActivityPredictions = wrongActivityPredictions[wrongActivityPredictions.GT=='other']
# # import matplotlib.pyplot as plt
# # plt.rcParams['figure.figsize'] = (16.0, 12.0)
# # plt.style.use('ggplot')
# # # plt.figure(figsize=(12, 8))
# # params = {'legend.fontsize': 'x-large',
# #           'figure.figsize': (15, 5),
# #          'axes.labelsize': 'x-large',
# #          'axes.titlesize':'x-large',
# #          'xtick.labelsize':'x-large',
# #          'ytick.labelsize':'x-large'}
# # plt.rcParams.update(params)
# # # # __________________ HOME PredictionError_start______________
# # # plt.xticks((np.arange(0, 24, step=1)))
# # # ax1 = (homeWrongActivityPredictions['start(sec)']/3600).plot(kind='hist', bins=60, density=True, alpha=0.5, label='prediction error of home activity', legend=True)
# # # ax1.set_xlabel(u'start (hour)')
# # # ax1.set_title('home prediction error based on start of the activity ')
# # # ax1.set_ylabel('error frequency')
# # # plt.savefig("D:/progress meeting/17June2020(Hans&Adam)/homePredictionError_start.png",dpi = 300)
# # # # ____________________WORK PredictionError_start _______________
# # # plt.figure()
# # # plt.xticks((np.arange(0, 24, step=1)))
# # # ax1 = (workWrongActivityPredictions['start(sec)']/3600).plot(kind='hist', bins=60, density=True, alpha=0.5, label='prediction error of work activity', legend=True)
# # # ax1.set_xlabel(u'start (hour)')
# # # ax1.set_title(u'work prediction error based on start of the activity ')
# # # ax1.set_ylabel(u'error frequency')
# # # plt.savefig("D:/progress meeting/17June2020(Hans&Adam)/workPredictionError_start.png",dpi = 300)
# # # ____________________OTHER PredictionError_start _______________
# # plt.figure()
# # plt.xticks((np.arange(0, 24, step=1)))
# # ax1 = (otherWrongActivityPredictions['start(sec)']/3600).plot(kind='hist', bins=60, density=True, alpha=0.5, label='prediction error of other activity', legend=True)
# # ax1.set_xlabel(u'start (hour)')
# # ax1.set_title(u'other prediction error based on start of the activity ')
# # ax1.set_ylabel(u'error frequency')
# # plt.savefig("D:/progress meeting/17June2020(Hans&Adam)/otherPredictionError_start.png",dpi = 300)
# # # ________________________________ STAY PredictionError_start___________________
# # clusterData =
# #
# # # # ____________________________________________________________
# # # testings:
# # # # import matplotlib.pyplot as plt
# # # # plt.rcParams['figure.figsize'] = (16.0, 12.0)
# # # # plt.style.use('ggplot')
# # # # plt.figure(figsize=(12, 8))
# # # # plt.xticks((np.arange(0, 24, step=1)))
# # # # (anchorrrrrLoc['duration(sec)']/3600).plot(kind='hist', bins=96, density=True, alpha=0.5, label='Gaussian Kernel Density Estimation', legend=True)
# # # # not identified stay locations:
# # # # id=173, end_time="18:48:19" link="7038258_0" start_time="18:17:33" type="leisure"
# # # # id="17308", end_time="15:07:41" link="7400277_0" start_time="14:36:26" type="sozializing"
# # #

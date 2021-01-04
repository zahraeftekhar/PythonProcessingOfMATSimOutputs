import pandas as pd
from scipy.stats import gaussian_kde
import time
import numpy as np

# snapDataSplit = pd.read_csv('D:/ax/gis/locationMappingToMezuroZones/finalInputPython/finalInputPython.CSV')  # 38.5 sec to load 4G data

#____________________________________ clustering records ________________________________________
files = range(0,10) ###### fixme warning: 0 is not in range because we have it but generally it should be
interval = 30 #todo: please enter the interval between snapData records
startTime = time.time()
for q in files: #27 min for each file
    # snapDataSplit = pd.read_csv("D:/ax/gis/completePLUdata_30sec/completePLUdata_30sec_{number}.csv".format(number = q)) #38.5 sec to load 4G data
    snapDataSplit = pd.read_csv("/data/zahraeftekhar/research_temporal/completePLUdata_30sec/completePLUdata_30sec_{number}.csv".format(number = q)) #38.5 sec to load 4G data
    snapDataSplit[['mzr_id', 'VEHICLE']] = snapDataSplit[['mzr_id', 'VEHICLE']].astype(int)
    snapDataSplit = snapDataSplit.sort_values(by=["VEHICLE","TIME"])
    vehicleIDs = snapDataSplit.VEHICLE.unique()
    vehicleIDs = np.sort(vehicleIDs)
    clusterData = pd.DataFrame()
    mmm=1
    i=0
    for i in (range(len(vehicleIDs))):#len(vehicleIDs)-1 #1962 sec for one file of 3200 user
        if i==mmm*100:
            print('{percentage} percent____{duration} sec'.format(percentage = i/len(vehicleIDs), duration = time.time()-startTime))
            mmm +=1
        testRecords = snapDataSplit[snapDataSplit.VEHICLE == vehicleIDs[i]]
        # testRecords['time'] = pd.to_timedelta(testRecords['TIME'],unit='s')

        # _____******
        snapDataSplit = snapDataSplit.drop(index=range(len(testRecords)))
        # if i==57:
        #     print(vehicleIDs[i])
        snapDataSplit = snapDataSplit.reset_index(drop=True)
        # _____*******

        clusIndex = [0]
        j = 0
        k = 1
        if len(testRecords)==1:
            newD = testRecords
            newD['duration(sec)'] =0
        else:
            while k < len(testRecords):
                # if i == 441:
                #     if k==46:
                #         print(i)
                #         print(k)
                while round(testRecords['EASTING'][j]*10**9)/(10**9)==round(testRecords['EASTING'][k]*10**9)/(10**9):
                    if (round(testRecords['NORTHING'][j]*10**8)/(10**8)==round(testRecords['NORTHING'][k]*10**8)/(10**8)):
                        k+=1
                        if(k == len(testRecords)):
                            break

                clusIndex += [k]
                j=k
                k+=1
            newD = pd.DataFrame(index=range(len(clusIndex)-1), columns=testRecords.columns)
            newD ['duration(sec)'] = None
            l=0
            for l in range(len(clusIndex)-1):

                if l==len(clusIndex)-2:
                    newD.iloc[l, -1] = testRecords.TIME[clusIndex[l + 1]-1] - testRecords.TIME[clusIndex[l]]+interval
                else:
                    newD.iloc[l,-1] = testRecords.TIME[clusIndex[l+1]]-testRecords.TIME[clusIndex[l]]
                newD.iloc[l,0:-1]=testRecords.iloc[clusIndex[l],:]
        # ____________________for when you do not have the last record ____________________
        # newD = newD.append(newD.loc[len(newD)-1,:])
        # newD = newD.reset_index(drop=True)
        # newD.loc[len(newD) - 1, 'TIME'] = newD.loc[len(newD) - 1, 'TIME']+newD.loc[len(newD) - 1, 'duration(sec)']
        # newD.loc[len(newD)-1,'duration(sec)'] =pd.to_timedelta(newD.loc[0,'TIME'],unit='s')- pd.to_timedelta(newD.loc[len(newD)-1,'TIME'],unit='s')+pd.to_timedelta('24:00:00')
        # newD.loc[len(newD) - 1, 'duration(sec)'] = to_seconds(newD.loc[len(newD)-1,'duration(sec)'])
        # _______________________________________________________________________________
        clusterData = clusterData.append(newD)
    print((int((time.time() - startTime)/60)), end=' min ____ end for file number {number}\n'.format(number=q))
    # clusterData.to_csv('D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/clusterPLU_30sec_{number}.CSV'.format(number = q), index=False, header=True)
    clusterData.to_csv('/data/zahraeftekhar/research_temporal/completePLUdata_30sec/clusterPLU_30sec/clusterPLU_30sec_{number}.CSV'.format(number = q), index=False, header=True)

#  _________________ modifying users in the boundry of files (splited records) _________________
for s in range(0,9):
    # cluster0 = pd.read_csv('D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/clusterPLU_30sec_{number}.CSV'.format(number=s))
    cluster0 = pd.read_csv('/data/zahraeftekhar/research_temporal/completePLUdata_30sec/clusterPLU_30sec/clusterPLU_30sec_{number}.CSV'.format(number=s))
    bound0 = cluster0.loc[len(cluster0)-1, :]
    # cluster1 = pd.read_csv('D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/clusterPLU_30sec_{number}.CSV'.format(number=s+1))
    cluster1 = pd.read_csv('/data/zahraeftekhar/research_temporal/completePLUdata_30sec/clusterPLU_30sec/clusterPLU_30sec_{number}.CSV'.format(number=s+1))
    bound1 = cluster1.loc[0, :]
    rec0 = cluster0[cluster0.VEHICLE == bound0.VEHICLE]
    rec0 = rec0.reset_index(drop=True)
    rec1 = cluster1[cluster1.VEHICLE == bound1.VEHICLE]
    if bound0.EASTING==bound1.EASTING and bound0.NORTHING==bound1.NORTHING:
        rec0.loc[len(rec0)-1,'duration(sec)']+=rec1.loc[0,'duration(sec)']
        rec1 = rec1.loc[1:,:]
        cluster0 = cluster0.append(rec1,ignore_index=True)
        cluster1 = cluster1.loc[(len(rec1)+1):,:]
    else:
        cluster0 = cluster0.append(rec1, ignore_index=True)
        cluster1 = cluster1.loc[len(rec1):, :]
    # cluster0.to_csv('D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/clusterPLU_30sec_{number}.CSV'.format(number=s), header=True,index=False)
    # cluster1.to_csv('D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/clusterPLU_30sec_{number}.CSV'.format(number=s+1), header=True,index=False)
    cluster0.to_csv('/data/zahraeftekhar/research_temporal/completePLUdata_30sec/clusterPLU_30sec/clusterPLU_30sec_{number}.CSV'.format(number=s), header=True,index=False)
    cluster1.to_csv('/data/zahraeftekhar/research_temporal/completePLUdata_30sec/clusterPLU_30sec/clusterPLU_30sec_{number}.CSV'.format(number=s+1), header=True,index=False)

# # _____________________________________________________________________________________________________________________
#_______________ location type identification _____________________________________________________
startTime = time.time()
print("************** interval is {inin} sec ***************".format(inin=interval))
seeds = range(101,126)
for seed in seeds:
    print("************** seed is {inin} sec ***************".format(inin=seed))
    # ________________________ reading data: Trainset _________________________
    # train_tripData = pd.read_csv(
    #     "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/ITERS/it.1/train_tripStarts_seed{ss}.csv".format(
    #         ss=seed))
    # train_activityData = pd.read_csv(
    #     "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/ITERS/it.1/train_activityStarts_seed{ss}.csv".format(
    #         ss=seed))
    # train_homeData = pd.read_csv(
    #     "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/ITERS/it.1/train_homeStarts_seed{ss}.CSV".format(
    #         ss=seed))
    # train_workData = pd.read_csv(
    #     "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/ITERS/it.1/train_workStarts_seed{ss}.CSV".format(
    #         ss=seed))
    # train_otherData = pd.read_csv(
    #     "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/ITERS/it.1/train_otherStarts_seed{ss}.CSV".format(
    #         ss=seed))



    train_tripData = pd.read_csv(
        "/data/zahraeftekhar/research_temporal/GTanalysis/train_tripStarts_seed{ss}.csv".format(
            ss=seed))
    train_activityData = pd.read_csv(
        "/data/zahraeftekhar/research_temporal/GTanalysis/train_activityStarts_seed{ss}.csv".format(
            ss=seed))
    train_homeData = pd.read_csv(
        "/data/zahraeftekhar/research_temporal/GTanalysis/train_homeStarts_seed{ss}.CSV".format(
            ss=seed))
    train_workData = pd.read_csv(
        "/data/zahraeftekhar/research_temporal/GTanalysis/train_workStarts_seed{ss}.CSV".format(
            ss=seed))
    train_otherData = pd.read_csv(
        "/data/zahraeftekhar/research_temporal/GTanalysis/train_otherStarts_seed{ss}.CSV".format(
            ss=seed))


    ntrip = len(train_tripData)
    nactivity = len(train_activityData)
    prior_activity = nactivity/(nactivity + ntrip)
    prior_trip = ntrip/(nactivity + ntrip)
    nHome = len(train_homeData)
    nWork = len(train_workData)
    nOther = len(train_otherData)
    prior_home = nHome/(nHome + nWork + nOther)
    prior_work = nWork/(nHome + nWork + nOther)
    prior_other = nOther/(nHome + nWork + nOther)
    files = range(0,10)
    r=0
    for r in files: #4 min for each file
        # clusterData = pd.read_csv('D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/clusterPLU_30sec_{number}.CSV'.format(number = r))
        clusterData = pd.read_csv('/data/zahraeftekhar/research_temporal/completePLUdata_30sec/clusterPLU_30sec/clusterPLU_30sec_{number}.CSV'.format(number = r))
        clusterData['location type'] = None
        clusterData['activity type'] = None
        clusterData['stay'] = np.log(prior_activity)+np.log( gaussian_kde(train_activityData['Duration(hour)']).pdf((clusterData['duration(sec)'].astype(int))/3600))+np.log( gaussian_kde(train_activityData['start(hour)']).pdf((clusterData['TIME'].astype(int))/3600))
        clusterData['pass-by'] = np.log(prior_trip)+np.log( gaussian_kde(train_tripData['Duration(hour)']).pdf((clusterData['duration(sec)'].astype(int))/3600))+np.log( gaussian_kde(train_tripData['start(hour)']).pdf((clusterData['TIME'].astype(int))/3600))
        clusterData['location type']=clusterData[['stay','pass-by']].idxmax(axis=1)
        clusterData['home'] = None
        clusterData['work'] = None
        clusterData['other'] = None
        vehicleIDs = clusterData.VEHICLE.unique().astype(int)
        anchorLocs = pd.DataFrame(columns=['VEHICLE','mzr_id','home','work', 'other','EASTING','NORTHING','start(sec)','duration(sec)'])
        i=0
        mmm=1
        for i in range(len(vehicleIDs)): #range(len(vehicleIDs))
            if i==mmm*1000:
                print('{percentage} percent____{duration} sec'.format(percentage = i/len(vehicleIDs), duration = time.time()-startTime))
                mmm +=1
            tempData = clusterData[clusterData.VEHICLE == vehicleIDs[i]]
            stayIndex = tempData[tempData['location type']=='stay'].index
            anchors = pd.DataFrame(index=stayIndex, columns=['VEHICLE','mzr_id','home','work', 'other','EASTING','NORTHING','start(sec)','duration(sec)'])
            anchors.loc[:,'VEHICLE'] = tempData.loc[stayIndex[0],'VEHICLE']

            for n in range(len(stayIndex)):
                clusterData.loc[stayIndex[n],['home']] = np.log(prior_home) + np.log(
                    gaussian_kde(train_homeData['Duration(hour)']).pdf((clusterData.loc[stayIndex[n],['duration(sec)']].astype(int))/3600)) + np.log(
                    gaussian_kde(train_homeData['start(hour)']).pdf((clusterData.loc[stayIndex[n],['TIME']].astype(int))/3600))
                clusterData.loc[stayIndex[n],['work']] = np.log(prior_work) + np.log(
                    gaussian_kde(train_workData['Duration(hour)']).pdf((clusterData.loc[stayIndex[n],['duration(sec)']].astype(int))/3600)) + np.log(
                    gaussian_kde(train_workData['start(hour)']).pdf((clusterData.loc[stayIndex[n],['TIME']].astype(int))/3600))
                clusterData.loc[stayIndex[n],['other']] = np.log(prior_other) + np.log(
                    gaussian_kde(train_otherData['Duration(hour)']).pdf((clusterData.loc[stayIndex[n],['duration(sec)']].astype(int))/3600)) + np.log(
                    gaussian_kde(train_otherData['start(hour)']).pdf((clusterData.loc[stayIndex[n],['TIME']].astype(int))/3600))
                clusterData.loc[stayIndex[n],['activity type']] = pd.to_numeric(clusterData[['home', 'work','other']].iloc[stayIndex[n]]).idxmax(axis=1)
                anchors.loc[stayIndex[n], 'EASTING'] = tempData.loc[stayIndex[n], 'EASTING']
                anchors.loc[stayIndex[n], 'NORTHING'] = tempData.loc[stayIndex[n], 'NORTHING']
                anchors.loc[stayIndex[n], 'mzr_id'] = tempData.loc[stayIndex[n], 'mzr_id']
                anchors.loc[stayIndex[n], 'start(sec)'] = tempData.loc[stayIndex[n], 'TIME']
                anchors.loc[stayIndex[n], 'duration(sec)'] = tempData.loc[stayIndex[n], 'duration(sec)']
                if clusterData.loc[stayIndex[n],'activity type'] == 'home':
                    anchors.loc[stayIndex[n],'home'] = 1
                    anchors.loc[stayIndex[n], 'work'] = 0
                    anchors.loc[stayIndex[n], 'other'] = 0

                elif clusterData.loc[stayIndex[n],'activity type'] == 'work':
                    anchors.loc[stayIndex[n],'home'] = 0
                    anchors.loc[stayIndex[n], 'work'] = 1
                    anchors.loc[stayIndex[n], 'other'] = 0
                else:
                    anchors.loc[stayIndex[n],'home'] = 0
                    anchors.loc[stayIndex[n], 'work'] = 0
                    anchors.loc[stayIndex[n], 'other'] = 1
            anchorLocs = anchorLocs.append(anchors)
        anchorLocs['activity'] = (anchorLocs[['home','work', 'other']]).astype(int).idxmax(axis=1)
        anchorLocs = anchorLocs.drop(columns = ['home','work', 'other'], axis = 1)
        print((int((time.time() - startTime) / 60)), end=' min ____ end for file number {number}\n'.format(number=r))
        # anchorLocs.to_csv('D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/anchorLocsPLU_30sec_seed{ss}_{number}.CSV'.format(number=r,ss=seed), header=True,index=False)
        # clusterData.to_csv('D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/identifiedClusters/identified_clusterPLU_30sec_seed{ss}_{number}.CSV'.format(number=r,ss=seed), header=True,index=False)
        anchorLocs.to_csv(
            '/data/zahraeftekhar/research_temporal/completePLUdata_30sec/clusterPLU_30sec/anchorLocsPLU_30sec_seed{ss}_{number}.CSV'.format(number=r,
                                                                                                                ss=seed),
            header=True, index=False)
        clusterData.to_csv(
            '/data/zahraeftekhar/research_temporal/completePLUdata_30sec/clusterPLU_30sec/identifiedClusters/identified_clusterPLU_30sec_seed{ss}_{number}.CSV'.format(
                number=r, ss=seed), header=True, index=False)
# _______________________reconsidering activity type based on home loc_________________
seeds = range(101, 126)
startTime = time.time()
for seed in seeds:
    print("************** seed is {inin} and time is {tt} min***************".format(inin=seed,tt=round((time.time() - startTime) / 60)))
    anchorLocs = pd.DataFrame()
    for t in range(0,10):
        # anchor = pd.read_csv('D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/anchorLocsPLU_30sec_seed{ss}_{number}.CSV'.format(number = t))
        anchor = pd.read_csv('/data/zahraeftekhar/research_temporal/completePLUdata_30sec/clusterPLU_30sec/anchorLocsPLU_30sec_seed{ss}_{number}.CSV'.format(ss=seed,number = t))
        anchorLocs= anchorLocs.append(anchor,ignore_index=True)
    vehicleIDs = anchorLocs['VEHICLE'].unique()
    for i in range(len(vehicleIDs)): #range(len(vehicleIDs))
            temp = (anchorLocs[anchorLocs.VEHICLE == vehicleIDs[i]])
            if len(anchorLocs.loc[temp.index[temp['activity'] == 'home'],])>0:
                homeEASTING = anchorLocs.loc[temp.index[temp['activity'] == 'home'],'EASTING'].reset_index(drop=True)[len(anchorLocs.loc[temp.index[temp['activity'] == 'home'],'EASTING'])-1]
                homeNORTHING = anchorLocs.loc[temp.index[temp['activity'] == 'home'],'NORTHING'].reset_index(drop=True)[len(anchorLocs.loc[temp.index[temp['activity'] == 'home'],'NORTHING'])-1]
                for j in temp.index.drop(temp.index[temp['activity'] == 'home']):
                    if ((float(anchorLocs.loc[j,'EASTING'])-float(homeEASTING))**2+ (float(anchorLocs.loc[j,'NORTHING'])-float(homeNORTHING))**2)**0.5<300:
                        anchorLocs.loc[j,'activity'] = 'home'
    # anchorLocs.to_csv("D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/anchorLocsPLU_30sec_seed{ss}.CSV".format(ss=seed),header=True,index=False)
    anchorLocs.to_csv("/data/zahraeftekhar/research_temporal/completePLUdata_30sec/clusterPLU_30sec/anchorLocsPLU_30sec_seed{ss}.CSV".format(ss=seed),header=True,index=False)
# __________ estimation OD matrix ___________  todo: put Amsterdam zoning map with data point on it in the report
# anchorLocs = pd.DataFrame()
# for t in range(0,10):
#     anchor = pd.read_csv('D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/anchorLocsPLU_30sec_{number}.CSV'.format(number = t))
#     anchorLocs= anchorLocs.append(anchor,ignore_index=True)
seeds = range(101, 126)
for seed in seeds:
    # anchorLocs = pd.read_csv("D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/anchorLocsPLU_30sec_seed{ss}.CSV".format(ss=seed))
    anchorLocs = pd.read_csv("/data/zahraeftekhar/research_temporal/completePLUdata_30sec/clusterPLU_30sec/anchorLocsPLU_30sec_seed{ss}.CSV".format(ss=seed))
    anchorLocs['end(sec)'] = anchorLocs['start(sec)']+anchorLocs['duration(sec)']
    # amsterdamMezuroZones = pd.read_csv('D:/ax/gis/locationMappingToMezuroZones/amsterdamMezuroZones.CSV', usecols=['mzr_id'])
    amsterdamMezuroZones = pd.read_csv('/data/zahraeftekhar/research_temporal/input_base/amsterdamMezuroZones.CSV', usecols=['mzr_id'])
    tazNames = amsterdamMezuroZones['mzr_id'] #5333 is also included but not in amsterdam so '0' zone represent it
    zoneZero = pd.Series(0)
    matrixRowColNames = tuple(zoneZero.append(tazNames))
    odsize=len(matrixRowColNames)
    ODstart = "06:30:00"
    ODend = "09:30:00"
    startTime_OD = pd.to_timedelta(ODstart)
    endTime_OD = pd.to_timedelta(ODend)
    ODMatrix_df = pd.DataFrame(np.zeros((odsize, odsize), dtype=np.int32),
                               columns=matrixRowColNames,
                               index=matrixRowColNames)  # creating empty OD matrix
    vehicleIDs = anchorLocs.VEHICLE.unique().astype(int)
    start_time = time.time()
    m=0
    for m in range(len(vehicleIDs)): #range(len(vehicleIDs))
        activityList = anchorLocs[anchorLocs.VEHICLE == vehicleIDs[m]]

        activityList = activityList.reset_index(drop=True)
        j=1
        while j < len(activityList):
            if j==len(activityList)-1:
                start_time1 = pd.to_timedelta(activityList.loc[j,'start(sec)'], unit='sec')
                end_time1 = pd.to_timedelta(activityList.loc[j,'end(sec)'], unit='sec')
                start_time2 = pd.to_timedelta(activityList.loc[0,'start(sec)'], unit='sec')+ pd.to_timedelta('24:00:00')
                endActivity = end_time1
                startNewActivity = start_time2
                if pd.to_timedelta('23:59:59')>=pd.to_timedelta(startTime_OD)>=pd.to_timedelta(start_time1):
                    startTime_OD =ODstart
                else:
                    startTime_OD = ODstart + pd.to_timedelta('24:00:00')
                if pd.to_timedelta('23:59:59')>=pd.to_timedelta(endTime_OD)>=pd.to_timedelta(start_time1):
                    endTime_OD =ODend
                else:
                    endTime_OD = ODend+ pd.to_timedelta('24:00:00')
            else:
                start_time1 = pd.to_timedelta(activityList.loc[j,'start(sec)'], unit='sec')
                end_time1 = pd.to_timedelta(activityList.loc[j,'end(sec)'], unit='sec')
                start_time2 = pd.to_timedelta(activityList.loc[(j + 1),'start(sec)'], unit='sec')
                endActivity = end_time1 # when using inconsistent timings : endActivity = min(end_time1, start_time2)
                startNewActivity = start_time2 # when using inconsistent timings : startNewActivity = max(end_time1, start_time2)
            if pd.to_timedelta(start_time1) <= pd.to_timedelta(startTime_OD) < pd.to_timedelta(startNewActivity):
                if endTime_OD <= endActivity:
                    break
                else:
                    while endTime_OD > endActivity:
                        origin = activityList.loc[j,'mzr_id']
                        if j == len(activityList) - 1:
                            destination = activityList.loc[(0),'mzr_id']
                        else:
                            destination = activityList.loc[(j+1), 'mzr_id']
                        ODMatrix_df[origin][destination] = ODMatrix_df[origin][destination] + 1
                        j += 1
                        if j > len(activityList) - 1: break
                        end_time1 = pd.to_timedelta(activityList.loc[j,'end(sec)'], unit='sec')
                        endActivity = end_time1 # when using inconsistent timings :endActivity = min(end_time1, start_time2)
                    break
                # continue

            j += 1
        # print(activityList.loc[0,'VEHICLE'], end='____')

    print(np.sum(np.sum(ODMatrix_df, axis=0))) #13264 for 30sec interval ***********
    ODMatrix_df.to_csv('/data/zahraeftekhar/research_temporal/completePLUdata_30sec/OD({a}-{b}_{c}-{d})_seed{ss}.CSV'.
                       format(a=ODstart[0:2], b= ODstart[3:5], c = ODend[0:2], d=ODend[3:5],ss=seed),header=True, index=True)
    # ODoriginal = pd.read_csv("D:/ax/OD(06-30_09-30).CSV")
    ODoriginal = pd.read_csv("/data/zahraeftekhar/research_temporal/output_base/OD(06-30_09-30).CSV")
    ODoriginal = ODoriginal.set_index('Unnamed: 0')
    print(np.sum(np.sum(ODoriginal, axis=0))) #13353 ************
# #
#
# # anchorrrrrLoc = anchorLocs
# # anchorrrrrLoc['start'] = pd.to_timedelta(anchorLocs.loc[:,'start(sec)'],unit='s')
# # anchorrrrrLoc['end'] = pd.to_timedelta(anchorLocs.loc[:,'end(sec)'],unit='s')
# # anchorrrrrLoc['ID'] = anchorrrrrLoc['VEHICLE']
# # anchorrrrrLoc['duration'] = pd.to_timedelta(anchorLocs.loc[:,'duration(sec)'],unit='s')
# _________________________ number of activities per user ______________________________
# anchorLocs = pd.DataFrame()
# for t in range(0,10):
#     anchor = pd.read_csv('D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/anchorLocsPLU_30sec_{number}.CSV'.format(number = t))
#     anchorLocs= anchorLocs.append(anchor,ignore_index=True)
# anchorLocs = pd.read_csv("D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/anchorLocsPLU_30sec.CSV")
# anchorLocs['end(sec)'] = anchorLocs['start(sec)']+anchorLocs['duration(sec)']
# from lxml import etree
# parser = etree.XMLParser(ns_clean=True, collect_ids=False)
# itemlistExperienced= etree.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples"
#                                "/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.experienced_plans_Nogeneric(all allowed).xml").getroot()
# vehicleIDs = anchorLocs['VEHICLE'].unique()
# i=0
# nActivity = pd.DataFrame(columns=['VEHICLE', 'GT', 'PLU'])
# # mmm=1
# for i in range(len(vehicleIDs)):
#     # if i==mmm*100:
#     #     print('{percentage} percent____{duration} sec'.format(percentage = i/len(vehicleIDs), duration = time.time()-startTime))
#     #     mmm +=1
#     nActivity.loc[i,'VEHICLE'] = vehicleIDs[i]
#     nActivity.loc[i,'PLU'] = len(anchorLocs.loc[anchorLocs.loc[:,'VEHICLE']==vehicleIDs[i],"VEHICLE"])
#     nActivity.loc[i,'GT'] = len(itemlistExperienced.find('person[@id="{id}"]'.format(id=vehicleIDs[i].astype(int))).findall('plan/activity'))-1
# nActivity['comparison'] = nActivity['PLU']==nActivity['GT']
# sum(nActivity['comparison'])/len(nActivity)
# sum(nActivity['PLU'])
# sum(nActivity['GT'])
# correctNActivity = nActivity[(nActivity['comparison']==1)].VEHICLE
# correctNActivity = correctNActivity.reset_index(drop=True)
# # _________________________________ activity type_________________________
# activityType = pd.DataFrame(columns=['VEHICLE', 'PLU','GT'])
#
# for i in range(len(correctNActivity)):
#     activityType.loc[i,'VEHICLE'] = correctNActivity[i]
#     testActivityPLU = []
#     for j in (anchorLocs.loc[anchorLocs.loc[:,'VEHICLE']==correctNActivity[i],"activity"]):
#         testActivityPLU+=[j]
#     activityType.at[i,'PLU'] = testActivityPLU
#     testActivityGT = []
#     for j in (((itemlistExperienced.find('person[@id="{id}"]'.format(id=correctNActivity[i].astype(int))).findall('plan/activity')[1:]))):
#         if j.get('type')!='home' and j.get('type')!='work':
#             testActivityGT += ['other']
#         else:
#             testActivityGT += [j.get('type')]
#     activityType.at[i, 'GT'] = testActivityGT
#
# activityType['comparison'] = activityType['PLU']==activityType['GT']
# sum(activityType['comparison'])/len(activityType)
# # ___________________________________________________________________
# activityType = pd.DataFrame(columns=['VEHICLE', 'PLU','GT'])
# testActivityGT = []
# testActivityPLU = []
# starts = []
# durations = []
# for i in range(len(correctNActivity)):
#     # activityType.loc[i,'VEHICLE'] = correctNActivity[i]
#     for j in (anchorLocs.loc[anchorLocs.loc[:,'VEHICLE']==correctNActivity[i],"activity"]):
#         testActivityPLU+=[j]
#     for l in range(len(anchorLocs.loc[anchorLocs.loc[:,'VEHICLE']==correctNActivity[i],"start(sec)"])):
#             starts+=[(anchorLocs.loc[anchorLocs.loc[:,'VEHICLE']==correctNActivity[i],"start(sec)"].reset_index(drop=True))[l]]
#             durations+=[anchorLocs.loc[anchorLocs.loc[:,'VEHICLE']==correctNActivity[i],"duration(sec)"].reset_index(drop=True)[l]]
#
#     # activityType.at[i,'PLU'] = testActivityPLU
#     for k in (((itemlistExperienced.find('person[@id="{id}"]'.format(id=correctNActivity[i].astype(int))).findall('plan/activity')[1:]))):
#         if k.get('type')!='home' and k.get('type')!='work':
#             testActivityGT += ['other']
#         else:
#             testActivityGT += [k.get('type')]
# activityType = pd.DataFrame(columns=['PLU','GT'])
# activityType['PLU'] = testActivityPLU
# activityType['GT'] = testActivityGT
# activityType['start(sec)'] = starts
# activityType['duration(sec)'] = durations
# activityType['comparison'] = activityType['PLU']==activityType['GT']
# # ________________________________________________________________________
# sum(activityType['comparison'])/len(activityType)
# wrongActivityPredictions = activityType[activityType.comparison==0]
# homeWrongActivityPredictions = wrongActivityPredictions[wrongActivityPredictions.GT=='home']
# workWrongActivityPredictions = wrongActivityPredictions[wrongActivityPredictions.GT=='work']
# otherWrongActivityPredictions = wrongActivityPredictions[wrongActivityPredictions.GT=='other']
# import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize'] = (16.0, 12.0)
# plt.style.use('ggplot')
# # plt.figure(figsize=(12, 8))
# params = {'legend.fontsize': 'x-large',
#           'figure.figsize': (15, 5),
#          'axes.labelsize': 'x-large',
#          'axes.titlesize':'x-large',
#          'xtick.labelsize':'x-large',
#          'ytick.labelsize':'x-large'}
# plt.rcParams.update(params)
# # # __________________ HOME PredictionError_start______________
# # plt.xticks((np.arange(0, 24, step=1)))
# # ax1 = (homeWrongActivityPredictions['start(sec)']/3600).plot(kind='hist', bins=60, density=True, alpha=0.5, label='prediction error of home activity', legend=True)
# # ax1.set_xlabel(u'start (hour)')
# # ax1.set_title('home prediction error based on start of the activity ')
# # ax1.set_ylabel('error frequency')
# # plt.savefig("D:/progress meeting/17June2020(Hans&Adam)/homePredictionError_start.png",dpi = 300)
# # # ____________________WORK PredictionError_start _______________
# # plt.figure()
# # plt.xticks((np.arange(0, 24, step=1)))
# # ax1 = (workWrongActivityPredictions['start(sec)']/3600).plot(kind='hist', bins=60, density=True, alpha=0.5, label='prediction error of work activity', legend=True)
# # ax1.set_xlabel(u'start (hour)')
# # ax1.set_title(u'work prediction error based on start of the activity ')
# # ax1.set_ylabel(u'error frequency')
# # plt.savefig("D:/progress meeting/17June2020(Hans&Adam)/workPredictionError_start.png",dpi = 300)
# # ____________________OTHER PredictionError_start _______________
# plt.figure()
# plt.xticks((np.arange(0, 24, step=1)))
# ax1 = (otherWrongActivityPredictions['start(sec)']/3600).plot(kind='hist', bins=60, density=True, alpha=0.5, label='prediction error of other activity', legend=True)
# ax1.set_xlabel(u'start (hour)')
# ax1.set_title(u'other prediction error based on start of the activity ')
# ax1.set_ylabel(u'error frequency')
# plt.savefig("D:/progress meeting/17June2020(Hans&Adam)/otherPredictionError_start.png",dpi = 300)
# # ________________________________ STAY PredictionError_start___________________
# clusterData =
#
# # # ____________________________________________________________
# # testings:
# # # import matplotlib.pyplot as plt
# # # plt.rcParams['figure.figsize'] = (16.0, 12.0)
# # # plt.style.use('ggplot')
# # # plt.figure(figsize=(12, 8))
# # # plt.xticks((np.arange(0, 24, step=1)))
# # # (anchorrrrrLoc['duration(sec)']/3600).plot(kind='hist', bins=96, density=True, alpha=0.5, label='Gaussian Kernel Density Estimation', legend=True)
# # # not identified stay locations:
# # # id=173, end_time="18:48:19" link="7038258_0" start_time="18:17:33" type="leisure"
# # # id="17308", end_time="15:07:41" link="7400277_0" start_time="14:36:26" type="sozializing"
# #

import pandas as pd
from scipy.stats import gaussian_kde
from shapely.geometry import Point
import time
from math import floor
import numpy as np
import os
import random
# ____________________________________ choosing records based on interval & clustering records ________________________________________
files = range(0,10) ###### fixme warning: 0 is not in range because we have it but generally it should be
# interval = 60#todo: please enter the interval between snapData records
intervals = [30,60,300,600,900,1200,1500,1800,2100,2400,2700,3600]
startTime = time.time()
q=0
# __________________________ generating snapshot for the specified interval _________________________________________________
for interval in intervals:
    print("************** interval is {inin} sec ***************".format(inin=interval))
    for q in files: # 18.6min for each file60,13.53min 300,20min 600, 20min900,16.25min for1800
        snapDataSplit = pd.read_csv("D:/ax/gis/completePLUdata_30sec/completePLUdata_30sec_{number}.csv".format(number = q)) #38.5 sec to load 4G data
        snapDataSplit[['mzr_id', 'VEHICLE']] = snapDataSplit[['mzr_id', 'VEHICLE']].astype(int)
        snapDataSplit = snapDataSplit.sort_values(by=["VEHICLE","TIME"])
        vehicleIDs = snapDataSplit.VEHICLE.unique()
        snapDataSplit_intervalFixed = pd.DataFrame(columns=['mzr_id', 'VEHICLE', 'TIME', 'EASTING', 'NORTHING'])
        i=1
        mmm=1
        for i in range(len(vehicleIDs)):
            if i==mmm*100:
                print('{percentage} percent____{duration} sec'.format(percentage = i/len(vehicleIDs), duration = time.time()-startTime))
                mmm +=1
            testRecords = snapDataSplit[snapDataSplit.VEHICLE == vehicleIDs[i]]
            snapDataSplit = snapDataSplit.drop(index=range(len(testRecords)))
            snapDataSplit = snapDataSplit.reset_index(drop=True)
            num = floor(len(testRecords)/(interval/30))
            random.seed(101)
            initialJump = random.choice((range(int(interval/30))))
            testRecords2 = testRecords.loc[[initialJump+(interval/30)*j for j in range(num)],:] # fixme: does this work (correct?)
            snapDataSplit_intervalFixed = snapDataSplit_intervalFixed.append(testRecords2)
        snapDataSplit = snapDataSplit_intervalFixed
        del snapDataSplit_intervalFixed
        # snapDataSplit = snapDataSplit.reset_index(drop=True)
        folderSave = "D:/ax/gis/completePLUdata_{inter}sec".format(inter=interval)
        path = 'path_to_my_folder'
        if not os.path.exists(folderSave):
            os.makedirs(folderSave)
        filename = 'completePLUdata_{inter}sec_{number}.csv'.format(number=q,inter=interval)
        filePathSave = os.path.join(folderSave,filename)
        snapDataSplit.to_csv(filePathSave, header=True, index=False)
        print('END of number{num}'.format(num = q))
       # # snapDataSplit.to_csv("D:/ax/gis/completePLUdata_{inter}sec/completePLUdata_{inter}sec_{number}.csv".format(number = q, inter = interval),header=True,index=False)
# # _______________________________________ clustering the records _________________________________________________
for interval in intervals:
    files = range(0,10) ###### fixme warning: 0 is not in range because we have it but generally it should be
    startTime = time.time()
    # q=0
    for q in files: #11 min for each file60...2.5min for interval300,1.4min for600, 1min for900, 0.5min1800
        snapDataSplit=pd.read_csv("D:/ax/gis/completePLUdata_{inter}sec/completePLUdata_{inter}sec_{number}.csv".format(number=q, inter=interval))
        # snapDataSplit = pd.read_csv("D:/ax/gis/completePLUdata_60sec/completePLUdata_60sec_{number}.csv".format(number = q)) #38.5 sec to load 4G data
        snapDataSplit[['mzr_id', 'VEHICLE']] = snapDataSplit[['mzr_id', 'VEHICLE']].astype(int)
        snapDataSplit = snapDataSplit.sort_values(by=["VEHICLE","TIME"])
        vehicleIDs = snapDataSplit.VEHICLE.unique()
        clusterData = pd.DataFrame()
        mmm=1
        for i in (range(len(vehicleIDs)-1)):#len(vehicleIDs) #1962 sec for one file of 3200 user
            if i==mmm*100:
                print('{percentage} percent____{duration} sec'.format(percentage = i/len(vehicleIDs), duration = time.time()-startTime))
                mmm +=1
            # if q==7:
            #     if i==2372:
            #         print(i)
            testRecords = snapDataSplit[snapDataSplit.VEHICLE == vehicleIDs[i]]
            snapDataSplit = snapDataSplit.drop(index=range(len(testRecords)))
            snapDataSplit = snapDataSplit.reset_index(drop=True)
            clusIndex = [0]
            j = 0
            k = 1
            while k < len(testRecords):
                while round(testRecords['EASTING'][j]*10**9)/(10**9)==round(testRecords['EASTING'][k]*10**9)/(10**9):
                    if (round(testRecords['NORTHING'][j]*10**7)/(10**7)==round(testRecords['NORTHING'][k]*10**7)/(10**7)):
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
            clusterData = clusterData.append(newD)
        print((int((time.time() - startTime)/60)), end=' min ____ end for file number {number}\n'.format(number=q))
        folderSave = "D:/ax/gis/completePLUdata_{inter}sec/clusterPLU_{inter}sec".format(inter=interval)
        if not os.path.exists(folderSave):
            os.makedirs(folderSave)
        filename = 'clusterPLU_{inter}sec_{number}.CSV'.format(number=q, inter=interval)
        filePathSave = os.path.join(folderSave, filename)
        clusterData.to_csv(filePathSave, index=False, header=True)
     #****** _________________ modifying users in the boundry of files (splited records) _________________
    for s in range(0,9):
        cluster0 = pd.read_csv('D:/ax/gis/completePLUdata_{inter}sec/clusterPLU_{inter}sec/clusterPLU_{inter}sec_{number}.CSV'.format(number=s,inter=interval))
        bound0 = cluster0.loc[len(cluster0)-1, :]
        cluster1 = pd.read_csv('D:/ax/gis/completePLUdata_{inter}sec/clusterPLU_{inter}sec/clusterPLU_{inter}sec_{number}.CSV'.format(number=s+1,inter=interval))
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
        cluster0.to_csv('D:/ax/gis/completePLUdata_{inter}sec/clusterPLU_{inter}sec/clusterPLU_{inter}sec_{number}.CSV'.format(number=s,inter=interval), header=True,index=False)
        cluster1.to_csv('D:/ax/gis/completePLUdata_{inter}sec/clusterPLU_{inter}sec/clusterPLU_{inter}sec_{number}.CSV'.format(number=s+1,inter=interval), header=True,index=False)
# # # _____________________________________________________________________________________________________________________
#__________________________________ location type identification _____________________________________________________
startTime = time.time()
#intervals = [3600]
# from _5_GTAnalysis_userSampling import train_activityData, train_tripData, train_homeData, train_otherData,train_workData
for interval in intervals:
    print("************** interval is {inin} sec ***************".format(inin=interval))
    seeds = range(101,126)
    for seed in seeds:
        print("************** seed is {inin} sec ***************".format(inin=seed))
        # ________________________ reading data: Trainset _________________________
        train_tripData = pd.read_csv(
            "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/ITERS/it.1/train_tripStarts_seed{ss}.csv".format(
                ss=seed))
        train_activityData = pd.read_csv(
            "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/ITERS/it.1/train_activityStarts_seed{ss}.csv".format(
                ss=seed))
        train_homeData = pd.read_csv(
            "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/ITERS/it.1/train_homeStarts_seed{ss}.CSV".format(
                ss=seed))
        train_workData = pd.read_csv(
            "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/ITERS/it.1/train_workStarts_seed{ss}.CSV".format(
                ss=seed))
        train_otherData = pd.read_csv(
            "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/ITERS/it.1/train_otherStarts_seed{ss}.CSV".format(
                ss=seed))
        # train_activityData = pd.read_csv('D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/train_activityData_0.01ratio.CSV')
        # train_tripData = pd.read_csv('D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/train_tripData_0.01ratio.CSV')
        # train_homeData = pd.read_csv('D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/train_homeData_0.01ratio.CSV')
        # train_workData = pd.read_csv('D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/train_workData_0.01ratio.CSV')
        # train_otherData = pd.read_csv('D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/train_otherData_0.01ratio.CSV')
        ntrip = len(train_tripData)
        nactivity = len(train_activityData)
        prior_activity = nactivity/(nactivity + ntrip)
        prior_trip = ntrip/(nactivity + ntrip)
        ### from groundTruthAnalysis_testBayesianForActivity import train_homeData, train_workData, train_otherData
        nHome = len(train_homeData)
        nWork = len(train_workData)
        nOther = len(train_otherData)
        prior_home = nHome/(nHome + nWork + nOther)
        prior_work = nWork/(nHome + nWork + nOther)
        prior_other = nOther/(nHome + nWork + nOther)
        files = range(0,10)
        # bw1 = 0.1
        # bw2 = 0.5
        # r=8
        for r in files: #4 min for each file60,2min for300, 1.8minfor600,1.7min for900,1min for1800
            clusterData = pd.read_csv('D:/ax/gis/completePLUdata_{inter}sec/clusterPLU_{inter}sec/clusterPLU_{inter}sec_{number}.CSV'.format(number = r, inter=interval))
            clusterData['location type'] = None
            clusterData['activity type'] = None
            clusterData['stay'] = np.log(prior_activity)+np.log( gaussian_kde(train_activityData['Duration(hour)']).pdf((clusterData['duration(sec)'].astype(int))/3600))+np.log( gaussian_kde(train_activityData['start(hour)']).pdf((clusterData['TIME'].astype(int))/3600))
            clusterData['pass-by'] = np.log(prior_trip)+np.log( gaussian_kde(train_tripData['Duration(hour)']).pdf((clusterData['duration(sec)'].astype(int))/3600))+np.log( gaussian_kde(train_tripData['start(hour)']).pdf((clusterData['TIME'].astype(int))/3600))
            clusterData['location type']=clusterData[['stay','pass-by']].idxmax(axis=1)
            clusterData['home'] = None
            clusterData['work'] = None
            clusterData['other'] = None
            vehicleIDs = clusterData.VEHICLE.unique().astype(int)
            # from groundTruthAnalysis_testBayesianForActivity import train_otherData, train_homeData, train_workData,nOther,nWork,nHome
            # train_homeData.to_csv('D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/train_homeData_0.01ratio.CSV', index=False, header=True)
            # train_workData.to_csv('D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/train_workData_0.01ratio.CSV', index=False, header=True)
            # train_otherData.to_csv('D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/train_otherData_0.01ratio.CSV', index=False, header=True)
            anchorLocs = pd.DataFrame(columns=['VEHICLE','mzr_id','home','work', 'other','EASTING','NORTHING','start(sec)','duration(sec)'])
            i=0
            mmm=1
            for i in range(len(vehicleIDs)): #range(len(vehicleIDs)) # problem with id= '160047'and '164302' ... skip it (boundry error)
                if i==mmm*1000:
                    print('{percentage} percent____{duration} sec'.format(percentage = i/len(vehicleIDs), duration = time.time()-startTime))
                    mmm +=1
                tempData = clusterData[clusterData.VEHICLE == vehicleIDs[i]]
                stayIndex = tempData[tempData['location type']=='stay'].index
                if len(stayIndex)>1:
                    anchors = pd.DataFrame(index=stayIndex, columns=['VEHICLE','mzr_id','home','work', 'other','EASTING','NORTHING','start(sec)','duration(sec)'])
                    anchors.loc[:,'VEHICLE'] = tempData.loc[stayIndex[0],'VEHICLE']

                    for n in range(len(stayIndex)):

                        # _____________ to extend the algorithm for labeling _______________________
                        # if(n>0):
                        #     if clusterData.loc[stayIndex[n-1],'EASTING'] in anchors.loc[stayIndex[:], 'EASTING'].values:
                        #         if clusterData.loc[stayIndex[n-1],'NORTHING'] in anchors.loc[stayIndex[:], 'NORTHING'].values:
                        #             copiedvalues = (anchors[anchors.loc[stayIndex[:], 'EASTING'].isin([clusterData.loc[stayIndex[n - 1], 'EASTING']])][['home', 'work', 'other']])
                        #             clusterData.loc[stayIndex[n],['home', 'work', 'other']] = copiedvalues.values
                        #             clusterData.loc[stayIndex[n], ['activity type']] = clusterData.loc[stayIndex[n],['home', 'work', 'other']].idxmax()
                        # _______________________________________________________________________


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
            anchorLocs.to_csv('D:/ax/gis/completePLUdata_{inter}sec/clusterPLU_{inter}sec/anchorLocsPLU_{inter}sec_seed{ss}_{number}.CSV'.format(number=r, inter=interval,ss=seed), header=True,index=False)
            folderSave = "D:/ax/gis/completePLUdata_{inter}sec/clusterPLU_{inter}sec/identifiedClusters".format(inter=interval)
            if not os.path.exists(folderSave):
                os.makedirs(folderSave)
            filename = 'identified_clusterPLU_{inter}sec_seed{ss}_{number}.CSV'.format(number = r, inter=interval,ss=seed)
            filePathSave = os.path.join(folderSave, filename)
            clusterData.to_csv(filePathSave, index=False, header=True)
            # clusterData.to_csv('D:/ax/gis/completePLUdata_{inter}sec/clusterPLU_{inter}sec/identifiedClusters/identified_clusterPLU_{inter}sec_{number}.CSV'.format(number=r, inter=interval), header=True,index=False)
# _______________________reconsidering activity type based on home loc_________________
# intervals = [3600] #2100,1800,60,600,900,1200,300,2400,30,,1500,2700
startTime = time.time()
for interval in intervals:
    print("************** interval is {inin} sec ***************".format(inin=interval))
    seeds = range(101,126)
    for seed in seeds:
        print("************** interval:{inter}, seed is {inin} and time is {tt} min***************".format(inin=seed,inter=interval,tt=round((time.time()-startTime)/60)))
        anchorLocs = pd.DataFrame()
        for t in range(0,10):
            anchor = pd.read_csv('D:/ax/gis/completePLUdata_{inter}sec/clusterPLU_{inter}sec/anchorLocsPLU_{inter}sec_seed{ss}_{number}.CSV'.format(number=t, inter=interval,ss=seed))
            anchorLocs= anchorLocs.append(anchor,ignore_index=True)
        vehicleIDs = anchorLocs['VEHICLE'].unique()
        for i in range(len(vehicleIDs)): #range(len(vehicleIDs))
            # if i%5000==0:
            #     print("{perc}% of seed {ss} in interval {inter}___time is {tt} min ".format(perc= round(i/len(vehicleIDs)*100),ss=seed,inter=interval,tt=round((time.time()-startTime)/60)))
            temp = (anchorLocs[anchorLocs.VEHICLE == vehicleIDs[i]])
            if len(anchorLocs.loc[temp.index[temp['activity'] == 'home'],])>0:
                homeEASTING = anchorLocs.loc[temp.index[temp['activity'] == 'home'],'EASTING'].reset_index(drop=True)[len(anchorLocs.loc[temp.index[temp['activity'] == 'home'],'EASTING'])-1]
                homeNORTHING = anchorLocs.loc[temp.index[temp['activity'] == 'home'],'NORTHING'].reset_index(drop=True)[len(anchorLocs.loc[temp.index[temp['activity'] == 'home'],'NORTHING'])-1]
                for j in temp.index.drop(temp.index[temp['activity'] == 'home']):
                    if ((float(anchorLocs.loc[j,'EASTING'])-float(homeEASTING))**2+ (float(anchorLocs.loc[j,'NORTHING'])-float(homeNORTHING))**2)**0.5<300:
                        anchorLocs.loc[j,'activity'] = 'home'
        anchorLocs.to_csv('D:/ax/gis/completePLUdata_{inter}sec/clusterPLU_{inter}sec/anchorLocsPLU_{inter}sec_seed{ss}.CSV'.format( inter=interval,ss=seed),header=True,index=False)

# #
# #
# _____________________fixing start time of activities 2 min elapsed time for interval=60s________________________________________
# intervals=[3600]
for interval in intervals:
    print("************** interval is {inin} sec ***************".format(inin=interval))
    seeds = range(101,126)
    for seed in seeds:
        print("************** seed is {inin} sec ***************".format(inin=seed))
        anchorLocs = pd.DataFrame() #NOT NEEDED for interval less than 5min
        speed  = 50*1000/3600 #35:(0.43,0.53),33:(0.45,),30:(0.473,0.494),28:(0.485,0.485)
        anchorLocs = pd.read_csv('D:/ax/gis/completePLUdata_{inter}sec/clusterPLU_{inter}sec/anchorLocsPLU_{inter}sec_seed{ss}.CSV'.format( inter=interval,ss=seed))
        anchorLocs = anchorLocs.sort_values('VEHICLE')
        GTtimeData=pd.read_csv("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.trueTimes.csv")
        GTtimeData.VEHICLE = (GTtimeData.VEHICLE).astype(int)
        GTtimeData = GTtimeData.sort_values('VEHICLE')
        GTids = np.sort((GTtimeData['VEHICLE'].unique()).astype(int))
        anchorLocs_fixedStart = pd.DataFrame()
        anchorLocsNogeneric = pd.DataFrame()
        for i in range(len(GTids)):
            anchorLocsNogeneric = anchorLocsNogeneric.append(anchorLocs[anchorLocs['VEHICLE']==GTids[i]])
        vehicleIDs = anchorLocs['VEHICLE'].unique()
        del anchorLocs
        vehicleIDs=GTids
        # i= 14623 #14619
        startTime = time.time()
        # mmm=1
        for i in range(len(vehicleIDs)):
            # if i==mmm*100:
            #     print('{percentage} percent____{duration} sec'.format(percentage = i/len(vehicleIDs), duration = time.time()-startTime))
            #     mmm +=1
            testRecords = anchorLocsNogeneric[anchorLocsNogeneric.VEHICLE == vehicleIDs[i]]
            testRecords = testRecords.reset_index(drop=True)
            testRecords2 = testRecords
            # testRecords.loc[0,'start(sec)'] = min((testRecords.loc[-1,'start(sec)']+testRecords.loc[-1,'duration(sec)']
            #                                        +(((testRecords.loc[-1,'EASTING']-testRecords.loc[0,'EASTING'])**2+(testRecords.loc[-1,'NORTHING']-testRecords.loc[0,'NORTHING'])**2)**(0.5))/speed),testRecords.loc[0,'start(sec)'])
            if len(testRecords2)>0:
                if (testRecords.loc[len(testRecords)-1,'start(sec)']+testRecords.loc[len(testRecords)-1,'duration(sec)']+(((testRecords.loc[len(testRecords)-1,'EASTING']-testRecords.loc[0,'EASTING'])**2+(testRecords.loc[len(testRecords)-1,'NORTHING']-testRecords.loc[0,'NORTHING'])**2)**(0.5))/speed)-24*3600< testRecords.loc[0,'start(sec)']:
                    if testRecords.loc[0, 'start(sec)']-(testRecords.loc[len(testRecords) - 1, 'start(sec)'] + testRecords.loc[len(testRecords) - 1, 'duration(sec)']+ (((testRecords.loc[len(testRecords) - 1, 'EASTING'] - testRecords.loc[0, 'EASTING']) ** 2 +(testRecords.loc[len(testRecords) - 1, 'NORTHING'] - testRecords.loc[0, 'NORTHING']) ** 2) ** (0.5)) / speed)+24*3600  < interval:
                        testRecords2.loc[0, 'duration(sec)'] += testRecords.loc[0,'start(sec)']-(testRecords.loc[len(testRecords)-1,'start(sec)']+testRecords.loc[len(testRecords)-1,'duration(sec)']+(((testRecords.loc[len(testRecords)-1,'EASTING']-testRecords.loc[0,'EASTING'])**2+(testRecords.loc[len(testRecords)-1,'NORTHING']-testRecords.loc[0,'NORTHING'])**2)**(0.5))/speed)+24*3600
                        testRecords2.loc[0, 'start(sec)'] = (testRecords.loc[len(testRecords)-1,'start(sec)']+testRecords.loc[len(testRecords)-1,'duration(sec)']+(((testRecords.loc[len(testRecords)-1,'EASTING']-testRecords.loc[0,'EASTING'])**2+(testRecords.loc[len(testRecords)-1,'NORTHING']-testRecords.loc[0,'NORTHING'])**2)**(0.5))/speed)-24*3600
                else:
                    testRecords2.loc[0, 'duration(sec)'] += interval
                    testRecords2.loc[0, 'start(sec)'] -= interval
                j=1
                for j in range(1,len(testRecords)):
                    if (testRecords.loc[j-1, 'start(sec)'] + testRecords.loc[j-1, 'duration(sec)']
                        + (((testRecords.loc[j-1, 'EASTING'] - testRecords.loc[j, 'EASTING']) ** 2 +
                            (testRecords.loc[j-1, 'NORTHING'] - testRecords.loc[j, 'NORTHING']) ** 2) ** (0.5)) / speed) < \
                            testRecords.loc[j, 'start(sec)']:
                        if  testRecords.loc[j, 'start(sec)']-(testRecords.loc[j-1, 'start(sec)'] + testRecords.loc[j-1, 'duration(sec)']
                                    + (((testRecords.loc[j-1, 'EASTING'] - testRecords.loc[j, 'EASTING']) ** 2 +
                                        (testRecords.loc[j-1, 'NORTHING'] - testRecords.loc[j, 'NORTHING']) ** 2) ** (0.5)) / speed)\
                                <interval:

                            testRecords2.loc[j, 'duration(sec)'] +=  testRecords.loc[j, 'start(sec)']-(testRecords.loc[j-1, 'start(sec)'] + testRecords.loc[j-1, 'duration(sec)']
                                        + (((testRecords.loc[j-1, 'EASTING'] - testRecords.loc[j, 'EASTING']) ** 2 +
                                            (testRecords.loc[j-1, 'NORTHING'] - testRecords.loc[j, 'NORTHING']) ** 2) ** (0.5)) / speed)\

                            testRecords2.loc[j, 'start(sec)'] = (
                                        testRecords.loc[j-1, 'start(sec)'] + testRecords.loc[j-1, 'duration(sec)']
                                        + (((testRecords.loc[j-1, 'EASTING'] - testRecords.loc[j, 'EASTING']) ** 2 +
                                            (testRecords.loc[j-1, 'NORTHING'] - testRecords.loc[j, 'NORTHING']) ** 2) ** (0.5)) / speed)
                        else:
                            testRecords2.loc[j, 'duration(sec)']+=interval
                            testRecords2.loc[j, 'start(sec)']-=(interval)
                anchorLocs_fixedStart = anchorLocs_fixedStart.append(testRecords2)
                # print(pd.to_timedelta(testRecords.loc[:, 'start(sec)'], unit='s'))
                # print(pd.to_timedelta(testRecords2.loc[:, 'start(sec)'], unit='s'))

        anchorLocs_fixedStart = anchorLocs_fixedStart.reset_index(drop=True)
        anchorLocs_fixedStart.to_csv('D:/ax/gis/completePLUdata_{inter}sec/clusterPLU_{inter}sec/anchorLocs_fixedStart_{inter}sec_seed{ss}.CSV'.format(ss=seed,inter=interval),header=True,index=False)
# # print(len(anchorLocs_fixedStart[anchorLocs_fixedStart['start(sec)']==anchorLocsNogeneric['start(sec)']])/len(anchorLocs_fixedStart))
# # print(len(anchorLocs_fixedStart[anchorLocs_fixedStart['start(sec)']-anchorLocsNogeneric['start(sec)']==interval])/len(anchorLocs_fixedStart))
# # anchorLocs_fixedStart['end(sec)'] = anchorLocs['start(sec)']+anchorLocs['duration(sec)']
# # anchorLocs_fixedStart['start'] = pd.to_timedelta(anchorLocs_fixedStart['start(sec)'],unit='s')
# # anchorLocsNogeneric['start'] = pd.to_timedelta(anchorLocsNogeneric['start(sec)'],unit='s')
# # anchorLocs_fixedStart['duration'] = pd.to_timedelta(anchorLocs_fixedStart['duration(sec)'],unit='s')
# # anchorLocsNogeneric['duration'] = pd.to_timedelta(anchorLocsNogeneric['duration(sec)'],unit='s')
# anchorLocsNogeneric.to_csv('D:/ax/gis/completePLUdata_{inter}sec/clusterPLU_{inter}sec/anchorLocs_NoGeneric.CSV'.format(inter=interval),header=True,index=False)
# # # #  __________________________test accuracy of start fixing  ___________________________________
# # # # GTtimeData=pd.read_csv("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples"
# # # #                                "/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.trueTimes.csv") #3 min
# # # GTtimeData = GTtimeData.drop(columns = ['end'])
# # # GTtimeData['startPLU'] = None
# # # GTtimeData['durationPLU']=None
# # # GTtimeData['startFixed'] = None
# # # GTtimeData['durationFixed'] = None
# # for i in range(len(vehicleIDs)):
# #     if len(GTtimeData[GTtimeData.VEHICLE==vehicleIDs[i]])==len(anchorLocs_fixedStart[anchorLocs_fixedStart.VEHICLE==vehicleIDs[i]]):
# #         GTtimeData.loc[:,'startFixed'][GTtimeData.VEHICLE==vehicleIDs[i]] = ((anchorLocs_fixedStart.loc[:,'start(sec)'][anchorLocs_fixedStart.VEHICLE==vehicleIDs[i]])).values
# #         GTtimeData.loc[:,'durationFixed'][GTtimeData.VEHICLE==vehicleIDs[i]] = ((anchorLocs_fixedStart.loc[:,'duration(sec)'][anchorLocs_fixedStart.VEHICLE==vehicleIDs[i]])).values
# #         GTtimeData.loc[:,'startPLU'][GTtimeData.VEHICLE==vehicleIDs[i]] = ((anchorLocsNogeneric.loc[:,'start(sec)'][anchorLocsNogeneric.VEHICLE==vehicleIDs[i]])).values
# #         GTtimeData.loc[:,'durationPLU'][GTtimeData.VEHICLE==vehicleIDs[i]] = ((anchorLocsNogeneric.loc[:,'duration(sec)'][anchorLocsNogeneric.VEHICLE==vehicleIDs[i]])).values
# # GTtimeData['startPLU2'] = pd.to_timedelta(GTtimeData['startPLU'],unit='s')
# # GTtimeData['durationPLU2']=pd.to_timedelta(GTtimeData['durationPLU'],unit='s')
# # GTtimeData['startFixed2'] = pd.to_timedelta(GTtimeData['startFixed'],unit='s')
# # GTtimeData['durationFixed2'] = pd.to_timedelta(GTtimeData['durationFixed'],unit='s')
# # GTtimeData = GTtimeData.drop(columns=['startPLU', 'durationPLU','startFixed', 'durationFixed'])
# # test = GTtimeData[['VEHICLE','start','startPLU2','startFixed2']]
# # # # ___________________________GT Times _____________________________________________
# # # from lxml import etree
# # # start_time=time.time()
# # # itemlistExperienced= etree.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples"
# # #                                "/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.experienced_plans_Nogeneric(all allowed).xml").getroot().findall('person')
# # # GTtimeData = pd.DataFrame(columns=['VEHICLE','start','duration','type'])
# # # mmm = 1
# # # for m, person in enumerate(itemlistExperienced):
# # #     if m == mmm * 100:
# # #         print('{percentage} percent____{duration} sec'.format(percentage=m / len(itemlistExperienced) * 100,
# # #                                                               duration=time.time() - start_time))
# # #         mmm += 1
# # #     # legList = itemlistExperienced[m].findall('plan/leg/route')
# # #     activityList = itemlistExperienced[m].findall('plan/activity')
# # #     i = 0
# # #     for i in range(1,len(activityList)-1):
# # #         trueloc = pd.DataFrame(columns=['VEHICLE','start','end','type'])
# # #         trueloc.loc[i-1, 'VEHICLE'] = person.get('id')
# # #         trueloc.loc[i - 1, 'start'] = activityList[i].get('start_time')
# # #         trueloc.loc[i - 1, 'end'] = activityList[i].get('end_time')
# # #         trueloc.loc[i - 1, 'type'] = activityList[i].get('type')
# # #     i+=1
# # #     trueloc.loc[i - 1, 'VEHICLE'] = person.get('id')
# # #     trueloc.loc[i - 1, 'start'] = activityList[i].get('start_time')
# # #     trueloc.loc[i - 1, 'end'] = pd.to_timedelta(activityList[0].get('end_time'))+pd.to_timedelta('24:00:00')
# # #     trueloc.loc[i - 1, 'type'] = activityList[i].get('type')
# # #     GTtimeData = GTtimeData.append(trueloc)
# # # GTtimeData.duration = pd.to_timedelta(GTtimeData.end)-pd.to_timedelta(GTtimeData.start)
# # # GTtimeData.to_csv("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples"
# # #                                "/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.trueTimes.csv", header=True, index=False)
# # # __________________________________________________________________________________
# #
# #
# __________ estimation OD matrix ___________  todo: put Amsterdam zoning map with data point on it in the report
# intervals = [2400]#1500,1800,2100,2400
for interval in intervals:
    print("************** interval is {inin} sec ***************".format(inin=interval))
    seeds = range(101, 126)
    for seed in seeds:
        anchorLocs = pd.read_csv('D:/ax/gis/completePLUdata_{inter}sec/clusterPLU_{inter}sec/anchorLocs_fixedStart_{inter}sec_seed{ss}.CSV'.format(ss=seed,inter=interval), usecols=['VEHICLE', 'mzr_id', 'EASTING', 'NORTHING', 'start(sec)',
               'duration(sec)','activity'])
        print(min(anchorLocs['duration(sec)']))
        anchorLocs['end(sec)'] = anchorLocs['start(sec)']+anchorLocs['duration(sec)']
        amsterdamMezuroZones = pd.read_csv('D:/ax/gis/locationMappingToMezuroZones/amsterdamMezuroZones.CSV', usecols=['mzr_id'])
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
        ODMatrix_home = pd.DataFrame(np.zeros((odsize, odsize), dtype=np.int32),
                                   columns=matrixRowColNames,
                                   index=matrixRowColNames)
        ODMatrix_work = pd.DataFrame(np.zeros((odsize, odsize), dtype=np.int32),
                                   columns=matrixRowColNames,
                                   index=matrixRowColNames)
        ODMatrix_other = pd.DataFrame(np.zeros((odsize, odsize), dtype=np.int32),
                                   columns=matrixRowColNames,
                                   index=matrixRowColNames)
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
                    from duration import to_seconds
                    from _datetime import datetime
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
                                if activityList.loc[(0),'activity']=='home':
                                    ODMatrix_home[origin][destination]=ODMatrix_home[origin][destination] + 1
                                elif activityList.loc[(0),'activity']== 'work':
                                    ODMatrix_work[origin][destination] = ODMatrix_work[origin][destination] + 1
                                else:
                                    ODMatrix_other[origin][destination] = ODMatrix_other[origin][destination] + 1

                            else:
                                destination = activityList.loc[(j+1), 'mzr_id']
                                if activityList.loc[(j+1),'activity']=='home':
                                    ODMatrix_home[origin][destination]=ODMatrix_home[origin][destination] + 1
                                elif activityList.loc[(j+1),'activity']== 'work':
                                    ODMatrix_work[origin][destination] = ODMatrix_work[origin][destination] + 1
                                else:
                                    ODMatrix_other[origin][destination] = ODMatrix_other[origin][destination] + 1
                            ODMatrix_df[origin][destination] = ODMatrix_df[origin][destination] + 1
                            j += 1
                            if j > len(activityList) - 1: break
                            end_time1 = pd.to_timedelta(activityList.loc[j,'end(sec)'], unit='sec')
                            endActivity = end_time1 # when using inconsistent timings :endActivity = min(end_time1, start_time2)
                        break
                    # continue

                j += 1
            # print(activityList.loc[0,'VEHICLE'], end='____')
        print(seed)
        print(np.sum(np.sum(ODMatrix_df, axis=0))) #13264 for 30sec interval
        ODMatrix_df.to_csv('D:/ax/gis/completePLUdata_{inter}sec/OD({a}-{b}_{c}-{d})_seed{ss}.CSV'.
                           format(inter = interval,a=ODstart[0:2], b= ODstart[3:5], c = ODend[0:2], d=ODend[3:5],ss=seed),header=True, index=True)
        ODMatrix_home.to_csv('D:/ax/gis/completePLUdata_{inter}sec/OD({a}-{b}_{c}-{d})_home_seed{ss}.CSV'.
                           format(inter = interval,a=ODstart[0:2], b= ODstart[3:5], c = ODend[0:2], d=ODend[3:5],ss=seed),header=True, index=True)
        ODMatrix_work.to_csv('D:/ax/gis/completePLUdata_{inter}sec/OD({a}-{b}_{c}-{d})_work_seed{ss}.CSV'.
                           format(inter = interval,a=ODstart[0:2], b= ODstart[3:5], c = ODend[0:2], d=ODend[3:5],ss=seed),header=True, index=True)
        ODMatrix_other.to_csv('D:/ax/gis/completePLUdata_{inter}sec/OD({a}-{b}_{c}-{d})_other_seed{ss}.CSV'.
                           format(inter = interval,a=ODstart[0:2], b= ODstart[3:5], c = ODend[0:2], d=ODend[3:5],ss=seed),header=True, index=True)

        ODoriginal = pd.read_csv('D:/ax/OD({a}-{b}_{c}-{d}).CSV'.
                           format(inter = interval,a=ODstart[0:2], b= ODstart[3:5], c = ODend[0:2], d=ODend[3:5]))
        ODoriginal = ODoriginal.set_index('Unnamed: 0')
        print(np.sum(np.sum(ODoriginal, axis=0))) #13353 ************
# # #
# # #
# # # # anchorrrrrLoc = anchorLocs
# # # # anchorrrrrLoc['start'] = pd.to_timedelta(anchorLocs.loc[:,'start(sec)'],unit='s')
# # # # anchorrrrrLoc['end'] = pd.to_timedelta(anchorLocs.loc[:,'end(sec)'],unit='s')
# # # # anchorrrrrLoc['ID'] = anchorrrrrLoc['VEHICLE']
# # # # anchorrrrrLoc['duration'] = pd.to_timedelta(anchorLocs.loc[:,'duration(sec)'],unit='s')
# # # # # _________________________ number of activities per user ______________________________
# # # # anchorLocs = pd.DataFrame()
# # # # for t in range(0,10):
# # # #     anchor = pd.read_csv('D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/anchorLocsPLU_30sec_{number}.CSV'.format(number = t))
# # # #     anchorLocs= anchorLocs.append(anchor,ignore_index=True)
# # # # anchorLocs['end(sec)'] = anchorLocs['start(sec)']+anchorLocs['duration(sec)']
# # # # from lxml import etree
# # # # parser = etree.XMLParser(ns_clean=True, collect_ids=False)
# # # # itemlistExperienced= etree.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples"
# # # #                                "/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.experienced_plans_Nogeneric(all allowed).xml").getroot()
# # # # vehicleIDs = anchorLocs['VEHICLE'].unique()
# # # # i=0
# # # # nActivity = pd.DataFrame(columns=['VEHICLE', 'GT', 'PLU'])
# # # # # mmm=1
# # # # for i in range(len(vehicleIDs)):
# # # #     # if i==mmm*100:
# # # #     #     print('{percentage} percent____{duration} sec'.format(percentage = i/len(vehicleIDs), duration = time.time()-startTime))
# # # #     #     mmm +=1
# # # #     nActivity.loc[i,'VEHICLE'] = vehicleIDs[i]
# # # #     nActivity.loc[i,'PLU'] = len(anchorLocs.loc[anchorLocs.loc[:,'VEHICLE']==vehicleIDs[i],"VEHICLE"])
# # # #     nActivity.loc[i,'GT'] = len(itemlistExperienced.find('person[@id="{id}"]'.format(id=vehicleIDs[i].astype(int))).findall('plan/activity'))-1
# # # # nActivity['comparison'] = nActivity['PLU']==nActivity['GT']
# # # # sum(nActivity['comparison'])/len(nActivity)
# # # # sum(nActivity['PLU'])
# # # # sum(nActivity['GT'])
# # # # correctNActivity = nActivity[(nActivity['comparison']==1)].VEHICLE
# # # # correctNActivity = correctNActivity.reset_index(drop=True)
# # # # # # # _________________________________ activity type_________________________
# # # # # # activityType = pd.DataFrame(columns=['VEHICLE', 'PLU','GT'])
# # # # # #
# # # # # for i in range(len(correctNActivity)):
# # # # #     activityType.loc[i,'VEHICLE'] = correctNActivity[i]
# # # # #     testActivityPLU = []
# # # # #     for j in (anchorLocs.loc[anchorLocs.loc[:,'VEHICLE']==correctNActivity[i],"activity"]):
# # # # #         testActivityPLU+=[j]
# # # # #     activityType.at[i,'PLU'] = testActivityPLU
# # # # #     testActivityGT = []
# # # # #     for j in (((itemlistExperienced.find('person[@id="{id}"]'.format(id=correctNActivity[i].astype(int))).findall('plan/activity')[1:]))):
# # # # #         if j.get('type')!='home' and j.get('type')!='work':
# # # # #             testActivityGT += ['other']
# # # # #         else:
# # # # #             testActivityGT += [j.get('type')]
# # # # #     activityType.at[i, 'GT'] = testActivityGT
# # # # #
# # # # # activityType['comparison'] = activityType['PLU']==activityType['GT']
# # # # # sum(activityType['comparison'])/len(activityType)
# # # # ___________________________________________________________________
# # # activityType = pd.DataFrame(columns=['VEHICLE', 'PLU','GT'])
# # # testActivityGT = []
# # # testActivityPLU = []
# # # starts = []
# # # durations = []
# # # for i in range(len(correctNActivity)):
# # #     # activityType.loc[i,'VEHICLE'] = correctNActivity[i]
# # #     for j in (anchorLocs.loc[anchorLocs.loc[:,'VEHICLE']==correctNActivity[i],"activity"]):
# # #         testActivityPLU+=[j]
# # #     for l in range(len(anchorLocs.loc[anchorLocs.loc[:,'VEHICLE']==correctNActivity[i],"start(sec)"])):
# # #             starts+=[(anchorLocs.loc[anchorLocs.loc[:,'VEHICLE']==correctNActivity[i],"start(sec)"].reset_index(drop=True))[l]]
# # #             durations+=[anchorLocs.loc[anchorLocs.loc[:,'VEHICLE']==correctNActivity[i],"duration(sec)"].reset_index(drop=True)[l]]
# # #
# # #     # activityType.at[i,'PLU'] = testActivityPLU
# # #     for k in (((itemlistExperienced.find('person[@id="{id}"]'.format(id=correctNActivity[i].astype(int))).findall('plan/activity')[1:]))):
# # #         if k.get('type')!='home' and k.get('type')!='work':
# # #             testActivityGT += ['other']
# # #         else:
# # #             testActivityGT += [k.get('type')]
# # # activityType = pd.DataFrame(columns=['PLU','GT'])
# # # activityType['PLU'] = testActivityPLU
# # # activityType['GT'] = testActivityGT
# # # activityType['start(sec)'] = starts
# # # activityType['duration(sec)'] = durations
# # # activityType['comparison'] = activityType['PLU']==activityType['GT']
# # # # ________________________________________________________________________
# # # sum(activityType['comparison'])/len(activityType)
# # # wrongActivityPredictions = activityType[activityType.comparison==0]
# # # homeWrongActivityPredictions = wrongActivityPredictions[wrongActivityPredictions.GT=='home']
# # # workWrongActivityPredictions = wrongActivityPredictions[wrongActivityPredictions.GT=='work']
# # # otherWrongActivityPredictions = wrongActivityPredictions[wrongActivityPredictions.GT=='other']
# # # import matplotlib.pyplot as plt
# # # plt.rcParams['figure.figsize'] = (16.0, 12.0)
# # # plt.style.use('ggplot')
# # # # plt.figure(figsize=(12, 8))
# # # params = {'legend.fontsize': 'x-large',
# # #           'figure.figsize': (15, 5),
# # #          'axes.labelsize': 'x-large',
# # #          'axes.titlesize':'x-large',
# # #          'xtick.labelsize':'x-large',
# # #          'ytick.labelsize':'x-large'}
# # # plt.rcParams.update(params)
# # # # # __________________ HOME PredictionError_start______________
# # # # plt.xticks((np.arange(0, 24, step=1)))
# # # # ax1 = (homeWrongActivityPredictions['start(sec)']/3600).plot(kind='hist', bins=60, density=True, alpha=0.5, label='prediction error of home activity', legend=True)
# # # # ax1.set_xlabel(u'start (hour)')
# # # # ax1.set_title('home prediction error based on start of the activity ')
# # # # ax1.set_ylabel('error frequency')
# # # # plt.savefig("D:/progress meeting/17June2020(Hans&Adam)/homePredictionError_start.png",dpi = 300)
# # # # # ____________________WORK PredictionError_start _______________
# # # # plt.figure()
# # # # plt.xticks((np.arange(0, 24, step=1)))
# # # # ax1 = (workWrongActivityPredictions['start(sec)']/3600).plot(kind='hist', bins=60, density=True, alpha=0.5, label='prediction error of work activity', legend=True)
# # # # ax1.set_xlabel(u'start (hour)')
# # # # ax1.set_title(u'work prediction error based on start of the activity ')
# # # # ax1.set_ylabel(u'error frequency')
# # # # plt.savefig("D:/progress meeting/17June2020(Hans&Adam)/workPredictionError_start.png",dpi = 300)
# # # # ____________________OTHER PredictionError_start _______________
# # # plt.figure()
# # # plt.xticks((np.arange(0, 24, step=1)))
# # # ax1 = (otherWrongActivityPredictions['start(sec)']/3600).plot(kind='hist', bins=60, density=True, alpha=0.5, label='prediction error of other activity', legend=True)
# # # ax1.set_xlabel(u'start (hour)')
# # # ax1.set_title(u'other prediction error based on start of the activity ')
# # # ax1.set_ylabel(u'error frequency')
# # # plt.savefig("D:/progress meeting/17June2020(Hans&Adam)/otherPredictionError_start.png",dpi = 300)
# # # # ________________________________ STAY PredictionError_start___________________
# # # clusterData =
# # #
# # # # # ____________________________________________________________
# # # # testings:
# # # # # import matplotlib.pyplot as plt
# # # # # plt.rcParams['figure.figsize'] = (16.0, 12.0)
# # # # # plt.style.use('ggplot')
# # # # # plt.figure(figsize=(12, 8))
# # # # # plt.xticks((np.arange(0, 24, step=1)))
# # # # # (anchorrrrrLoc['duration(sec)']/3600).plot(kind='hist', bins=96, density=True, alpha=0.5, label='Gaussian Kernel Density Estimation', legend=True)
# # # # # not identified stay locations:
# # # # # id=173, end_time="18:48:19" link="7038258_0" start_time="18:17:33" type="leisure"
# # # # # id="17308", end_time="15:07:41" link="7400277_0" start_time="14:36:26" type="sozializing"
# # # #
































import pandas as pd
import numpy as np
import time

# _____________________fixing start time of activities 2 min elapsed time for interval=60s________________________________________
intervals = [60,300,600,900,1200,1500,1800,2100,2400,2700,3000,3300,3600,4500,5400,6300,7200]
# for interval in intervals:
    # print("************** interval is {inin} sec ***************".format(inin=interval))
    # seeds = range(101,126)
    # for seed in seeds:
        # print("************** seed is {inin} sec ***************".format(inin=seed))
        # anchorLocs = pd.DataFrame() #NOT NEEDED for interval less than 5min
        # speed  = 50*1000/3600 #35:(0.43,0.53),33:(0.45,),30:(0.473,0.494),28:(0.485,0.485)
        # # anchorLocs = pd.read_csv('D:/ax/gis/completePLUdata_{inter}sec/clusterPLU_{inter}sec/anchorLocsPLU_{inter}sec_seed{ss}.CSV'.format( inter=interval,ss=seed))
        # anchorLocs = pd.read_csv('/data/zahraeftekhar/research_temporal/completePLUdata_{inter}sec/clusterPLU_{inter}sec/anchorLocsPLU_{inter}sec_seed{ss}.CSV'.format( inter=interval,ss=seed))
        # anchorLocs = anchorLocs.sort_values('VEHICLE')
        # # # GTtimeData=pd.read_csv("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.trueTimes.csv")
        # # GTtimeData=pd.read_csv("/data/zahraeftekhar/research_temporal/1.trueTimes.csv")#???????????????????? todo: what is TrueTimes
        # # GTtimeData.VEHICLE = (GTtimeData.VEHICLE).astype(int)
        # # GTtimeData = GTtimeData.sort_values('VEHICLE')
        # # GTids = np.sort((GTtimeData['VEHICLE'].unique()).astype(int))
        # anchorLocs_fixedStart = pd.DataFrame()
        # # for i in range(len(GTids)):
        # #     anchorLocsNogeneric = anchorLocsNogeneric.append(anchorLocs[anchorLocs['VEHICLE']==GTids[i]])
        # vehicleIDs = anchorLocs['VEHICLE'].unique()
        # anchorLocsNogeneric = anchorLocs
        # del anchorLocs
        # # vehicleIDs=GTids
        # # i= 14623 #14619
        # startTime = time.time()
        # # mmm=1
        # for i in range(len(vehicleIDs)):
            # # if i==mmm*100:
            # #     print('{percentage} percent____{duration} sec'.format(percentage = i/len(vehicleIDs), duration = time.time()-startTime))
            # #     mmm +=1
            # testRecords = anchorLocsNogeneric[anchorLocsNogeneric.VEHICLE == vehicleIDs[i]]
            # testRecords = testRecords.reset_index(drop=True)
            # testRecords2 = testRecords
            # # testRecords.loc[0,'start(sec)'] = min((testRecords.loc[-1,'start(sec)']+testRecords.loc[-1,'duration(sec)']
            # #                                        +(((testRecords.loc[-1,'EASTING']-testRecords.loc[0,'EASTING'])**2+(testRecords.loc[-1,'NORTHING']-testRecords.loc[0,'NORTHING'])**2)**(0.5))/speed),testRecords.loc[0,'start(sec)'])
            # if len(testRecords2)>0:
                # if (testRecords.loc[len(testRecords)-1,'start(sec)']+testRecords.loc[len(testRecords)-1,'duration(sec)']+(((testRecords.loc[len(testRecords)-1,'EASTING']-testRecords.loc[0,'EASTING'])**2+(testRecords.loc[len(testRecords)-1,'NORTHING']-testRecords.loc[0,'NORTHING'])**2)**(0.5))/speed)-24*3600< testRecords.loc[0,'start(sec)']:
                    # if testRecords.loc[0, 'start(sec)']-(testRecords.loc[len(testRecords) - 1, 'start(sec)'] + testRecords.loc[len(testRecords) - 1, 'duration(sec)']+ (((testRecords.loc[len(testRecords) - 1, 'EASTING'] - testRecords.loc[0, 'EASTING']) ** 2 +(testRecords.loc[len(testRecords) - 1, 'NORTHING'] - testRecords.loc[0, 'NORTHING']) ** 2) ** (0.5)) / speed)+24*3600  < interval:
                        # testRecords2.loc[0, 'duration(sec)'] += testRecords.loc[0,'start(sec)']-(testRecords.loc[len(testRecords)-1,'start(sec)']+testRecords.loc[len(testRecords)-1,'duration(sec)']+(((testRecords.loc[len(testRecords)-1,'EASTING']-testRecords.loc[0,'EASTING'])**2+(testRecords.loc[len(testRecords)-1,'NORTHING']-testRecords.loc[0,'NORTHING'])**2)**(0.5))/speed)+24*3600
                        # testRecords2.loc[0, 'start(sec)'] = (testRecords.loc[len(testRecords)-1,'start(sec)']+testRecords.loc[len(testRecords)-1,'duration(sec)']+(((testRecords.loc[len(testRecords)-1,'EASTING']-testRecords.loc[0,'EASTING'])**2+(testRecords.loc[len(testRecords)-1,'NORTHING']-testRecords.loc[0,'NORTHING'])**2)**(0.5))/speed)-24*3600
                # else:
                    # testRecords2.loc[0, 'duration(sec)'] += interval
                    # testRecords2.loc[0, 'start(sec)'] -= interval
                # j=1
                # for j in range(1,len(testRecords)):
                    # if (testRecords.loc[j-1, 'start(sec)'] + testRecords.loc[j-1, 'duration(sec)']
                        # + (((testRecords.loc[j-1, 'EASTING'] - testRecords.loc[j, 'EASTING']) ** 2 +
                            # (testRecords.loc[j-1, 'NORTHING'] - testRecords.loc[j, 'NORTHING']) ** 2) ** (0.5)) / speed) < \
                            # testRecords.loc[j, 'start(sec)']:
                        # if  testRecords.loc[j, 'start(sec)']-(testRecords.loc[j-1, 'start(sec)'] + testRecords.loc[j-1, 'duration(sec)']
                                    # + (((testRecords.loc[j-1, 'EASTING'] - testRecords.loc[j, 'EASTING']) ** 2 +
                                        # (testRecords.loc[j-1, 'NORTHING'] - testRecords.loc[j, 'NORTHING']) ** 2) ** (0.5)) / speed)\
                                # <interval:

                            # testRecords2.loc[j, 'duration(sec)'] +=  testRecords.loc[j, 'start(sec)']-(testRecords.loc[j-1, 'start(sec)'] + testRecords.loc[j-1, 'duration(sec)']
                                        # + (((testRecords.loc[j-1, 'EASTING'] - testRecords.loc[j, 'EASTING']) ** 2 +
                                            # (testRecords.loc[j-1, 'NORTHING'] - testRecords.loc[j, 'NORTHING']) ** 2) ** (0.5)) / speed)\

                            # testRecords2.loc[j, 'start(sec)'] = (
                                        # testRecords.loc[j-1, 'start(sec)'] + testRecords.loc[j-1, 'duration(sec)']
                                        # + (((testRecords.loc[j-1, 'EASTING'] - testRecords.loc[j, 'EASTING']) ** 2 +
                                            # (testRecords.loc[j-1, 'NORTHING'] - testRecords.loc[j, 'NORTHING']) ** 2) ** (0.5)) / speed)
                        # else:
                            # testRecords2.loc[j, 'duration(sec)']+=interval
                            # testRecords2.loc[j, 'start(sec)']-=(interval)
                # anchorLocs_fixedStart = anchorLocs_fixedStart.append(testRecords2)
                # # print(pd.to_timedelta(testRecords.loc[:, 'start(sec)'], unit='s'))
                # # print(pd.to_timedelta(testRecords2.loc[:, 'start(sec)'], unit='s'))

        # anchorLocs_fixedStart = anchorLocs_fixedStart.reset_index(drop=True)
        # # anchorLocs_fixedStart.to_csv('D:/ax/gis/completePLUdata_{inter}sec/clusterPLU_{inter}sec/anchorLocs_fixedStart_{inter}sec_seed{ss}.CSV'.format(ss=seed,inter=interval),header=True,index=False)
        # anchorLocs_fixedStart.to_csv(
            # '/data/zahraeftekhar/research_temporal/completePLUdata_{inter}sec/clusterPLU_{inter}sec/anchorLocs_fixedStart_{inter}sec_seed{ss}.CSV'.format(
                # ss=seed, inter=interval), header=True, index=False)

# __________ estimation OD matrix ___________  todo: put Amsterdam zoning map with data point on it in the report
for interval in intervals:
    print("************** interval is {inin} sec ***************".format(inin=interval))
    seeds = range(101, 126)
    for seed in seeds:
        # anchorLocs = pd.read_csv('D:/ax/gis/completePLUdata_{inter}sec/clusterPLU_{inter}sec/anchorLocs_fixedStart_{inter}sec_seed{ss}.CSV'.format(ss=seed,inter=interval), usecols=['VEHICLE', 'mzr_id', 'EASTING', 'NORTHING', 'start(sec)',
        #        'duration(sec)','activity'])
        # anchorLocs = pd.read_csv('/data/zahraeftekhar/research_temporal/completePLUdata_{inter}sec/clusterPLU_{inter}sec/anchorLocs_fixedStart_{inter}sec_seed{ss}.CSV'.format(ss=seed,inter=interval), usecols=['VEHICLE', 'mzr_id', 'EASTING', 'NORTHING', 'start(sec)',
               # 'duration(sec)','activity'])
        anchorLocs = pd.read_csv('/data/zahraeftekhar/research_temporal/completePLUdata_{inter}sec/clusterPLU_{inter}sec/anchorLocsPLU_{inter}sec_seed{ss}.CSV'.format(ss=seed,inter=interval), usecols=['VEHICLE', 'mzr_id', 'EASTING', 'NORTHING', 'start(sec)',
               'duration(sec)','activity'])
        print(min(anchorLocs['duration(sec)']))
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
        # ODMatrix_df.to_csv('D:/ax/gis/completePLUdata_{inter}sec/OD({a}-{b}_{c}-{d})_seed{ss}.CSV'.
        #                    format(inter = interval,a=ODstart[0:2], b= ODstart[3:5], c = ODend[0:2], d=ODend[3:5],ss=seed),header=True, index=True)
        ODMatrix_df.to_csv('/data/zahraeftekhar/research_temporal/completePLUdata_{inter}sec/OD({a}-{b}_{c}-{d})_seed{ss}.CSV'.
                           format(inter = interval,a=ODstart[0:2], b= ODstart[3:5], c = ODend[0:2], d=ODend[3:5],ss=seed),header=True, index=True)
        # ODMatrix_home.to_csv('D:/ax/gis/completePLUdata_{inter}sec/OD({a}-{b}_{c}-{d})_home_seed{ss}.CSV'.
        #                    format(inter = interval,a=ODstart[0:2], b= ODstart[3:5], c = ODend[0:2], d=ODend[3:5],ss=seed),header=True, index=True)
        # ODMatrix_work.to_csv('D:/ax/gis/completePLUdata_{inter}sec/OD({a}-{b}_{c}-{d})_work_seed{ss}.CSV'.
        #                    format(inter = interval,a=ODstart[0:2], b= ODstart[3:5], c = ODend[0:2], d=ODend[3:5],ss=seed),header=True, index=True)
        # ODMatrix_other.to_csv('D:/ax/gis/completePLUdata_{inter}sec/OD({a}-{b}_{c}-{d})_other_seed{ss}.CSV'.
        #                    format(inter = interval,a=ODstart[0:2], b= ODstart[3:5], c = ODend[0:2], d=ODend[3:5],ss=seed),header=True, index=True)

        # ODoriginal = pd.read_csv('D:/ax/OD({a}-{b}_{c}-{d}).CSV'.
        #                    format(inter = interval,a=ODstart[0:2], b= ODstart[3:5], c = ODend[0:2], d=ODend[3:5]))
        # ODoriginal = pd.read_csv('/data/zahraeftekhar/research_temporal/output_base/OD({a}-{b}_{c}-{d}).CSV'.
        #                    format(inter = interval,a=ODstart[0:2], b= ODstart[3:5], c = ODend[0:2], d=ODend[3:5]))
        # ODoriginal = ODoriginal.set_index('Unnamed: 0')
        # print(np.sum(np.sum(ODoriginal, axis=0))) #13353 ************
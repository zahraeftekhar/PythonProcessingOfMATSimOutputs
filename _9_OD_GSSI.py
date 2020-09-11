import pandas as pd
from scipy.stats import gaussian_kde
from shapely.geometry import Point
import time
from math import floor
import numpy as np
from statistics import mean
seeds=tuple(range(101,126))
intervals = tuple([30,60,300,600,900,1200,1500,1800,2100,2400,2700,3600])
amsterdamZones = pd.read_csv('D:/ax/gis/locationMappingToMezuroZones/CompleteAmsterdamMezuroZones.CSV')

# # ***************************** to be added ************************************
# tableGSSI = pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)
# tableSum = pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)
# tableSlope = pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)
# tableRsq = pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)
# #  ***************************** to be added ********************************

tableGSSI_home = pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)
tableSum_home = pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)
tableSlope_home = pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)
tableRsq_home = pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)


tableGSSI_work = pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)
tableSum_work = pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)
tableSlope_work = pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)
tableRsq_work = pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)


tableGSSI_other = pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)
tableSum_other = pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)
tableSlope_other = pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)
tableRsq_other= pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)


arrays = [amsterdamZones['gem_id'].append(pd.Series(0)),amsterdamZones['mzr_id'].append(pd.Series(0))]
tuples = sorted(list(zip(*arrays)))
tuples.sort(key = lambda x: x[1])
index = pd.MultiIndex.from_tuples(tuples, names=['gem_id','mzr_id'])

# # **********************to be added *************************
# GTOD = pd.read_csv('D:/ax/OD(06-30_09-30).CSV')
# GTOD = GTOD.set_index(GTOD['Unnamed: 0'], drop=True)
# GTOD = GTOD.iloc[:,1:]
# GTOD = GTOD.sort_index(axis = 1)
# GTOD = GTOD.sort_index(axis = 0)
# GTOD = pd.DataFrame(GTOD.values, index=index, columns=index)
# # ************** to be added *******************************

GTOD_home = pd.read_csv('D:/ax/OD(06-30_09-30)_home.CSV')
GTOD_home = GTOD_home.set_index(GTOD_home['Unnamed: 0'], drop=True)
GTOD_home = GTOD_home.iloc[:,1:]
GTOD_home = GTOD_home.sort_index(axis = 1)
GTOD_home = GTOD_home.sort_index(axis = 0)
GTOD_home = pd.DataFrame(GTOD_home.values, index=index, columns=index)
GTOD_work = pd.read_csv('D:/ax/OD(06-30_09-30)_work.CSV')
GTOD_work = GTOD_work.set_index(GTOD_work['Unnamed: 0'], drop=True)
GTOD_work = GTOD_work.iloc[:,1:]
GTOD_work = GTOD_work.sort_index(axis = 1)
GTOD_work = GTOD_work.sort_index(axis = 0)
GTOD_work = pd.DataFrame(GTOD_work.values, index=index, columns=index)
GTOD_other = pd.read_csv('D:/ax/OD(06-30_09-30)_other.CSV')
GTOD_other = GTOD_other.set_index(GTOD_other['Unnamed: 0'], drop=True)
GTOD_other = GTOD_other.iloc[:,1:]
GTOD_other = GTOD_other.sort_index(axis = 1)
GTOD_other = GTOD_other.sort_index(axis = 0)
GTOD_other = pd.DataFrame(GTOD_other.values, index=index, columns=index)
for interval in intervals:
    print("************** interval is {inin} sec ***************".format(inin=interval))
    seeds = range(101,126)
    for seed in seeds:
        print("************** seed is {inin} ***************".format(inin=seed))
        # # *******************to be added *****************************
        # cellOD = pd.read_csv('D:/ax/gis/completePLUdata_{inter}sec/OD(06-30_09-30)_seed{ss}.CSV'.format(inter = interval,ss=seed)) #todo: please enter the address of the PLU OD
        # cellOD = cellOD.set_index(cellOD['Unnamed: 0'], drop=True)
        # cellOD = cellOD.iloc[:,1:]
        # cellOD2 = cellOD.sort_index(axis = 1)
        # cellOD2 = cellOD2.sort_index(axis = 0)
        # PLUOD = pd.DataFrame(cellOD2.values, index=index, columns=index)
        # tableSum.loc[interval,seed] = np.sum(np.sum(PLUOD, axis=0))
        # # *****************to be added ************************************

        cellOD = pd.read_csv('D:/ax/gis/completePLUdata_{inter}sec/OD(06-30_09-30)_home_seed{ss}.CSV'.format(inter = interval,ss=seed)) #todo: please enter the address of the PLU OD
        cellOD = cellOD.set_index(cellOD['Unnamed: 0'], drop=True)
        cellOD = cellOD.iloc[:,1:]
        cellOD2 = cellOD.sort_index(axis = 1)
        cellOD2 = cellOD2.sort_index(axis = 0)
        PLUOD_home = pd.DataFrame(cellOD2.values, index=index, columns=index)
        tableSum_home.loc[interval,seed] = np.sum(np.sum(PLUOD_home, axis=0))


        cellOD = pd.read_csv('D:/ax/gis/completePLUdata_{inter}sec/OD(06-30_09-30)_work_seed{ss}.CSV'.format(inter = interval,ss=seed)) #todo: please enter the address of the PLU OD
        cellOD = cellOD.set_index(cellOD['Unnamed: 0'], drop=True)
        cellOD = cellOD.iloc[:,1:]
        cellOD2 = cellOD.sort_index(axis = 1)
        cellOD2 = cellOD2.sort_index(axis = 0)
        PLUOD_work = pd.DataFrame(cellOD2.values, index=index, columns=index)
        tableSum_work.loc[interval,seed] = np.sum(np.sum(PLUOD_work, axis=0))


        cellOD = pd.read_csv('D:/ax/gis/completePLUdata_{inter}sec/OD(06-30_09-30)_other_seed{ss}.CSV'.format(inter = interval,ss=seed)) #todo: please enter the address of the PLU OD
        cellOD = cellOD.set_index(cellOD['Unnamed: 0'], drop=True)
        cellOD = cellOD.iloc[:,1:]
        cellOD2 = cellOD.sort_index(axis = 1)
        cellOD2 = cellOD2.sort_index(axis = 0)
        PLUOD_other = pd.DataFrame(cellOD2.values, index=index, columns=index)
        tableSum_other.loc[interval,seed] = np.sum(np.sum(PLUOD_other, axis=0))
        #
        # # ****************************to be added ****************************
        # SSIMs = []
        # for i,gemrow in enumerate((amsterdamZones['gem_id'].append(pd.Series(0))).unique()):
        #     for j, gemcol in enumerate((amsterdamZones['gem_id'].append(pd.Series(0))).unique()):
        #         PLUwindow = PLUOD.loc[gemrow][gemcol]
        #         miuX = PLUwindow.values.mean()
        #         stX = PLUwindow.values.std()
        #         normPLUwindow = (PLUwindow-miuX)/stX
        #         GTwindow = GTOD.loc[gemrow][gemcol]
        #         miuY = GTwindow.values.mean()
        #         stY = GTwindow.values.std()
        #         normGTwindow = (GTwindow-miuY)/stY
        #         miuX=0
        #         miuY=0
        #         stX=1
        #         stY=1
        #         corXY = pd.Series(normPLUwindow.values.flatten()).corr(pd.Series(normGTwindow.values.flatten()),method='pearson')
        #         # corXY=np.cov(pd.Series(normPLUwindow.values.flatten()),pd.Series(normGTwindow.values.flatten())).mean()
        #         # corXY = np.cov(pd.Series(normPLUwindow.values.flatten()),pd.Series(normGTwindow.values.flatten())).mean()/(stX*stY)
        #         SSIMtest = ((2*miuX*miuY+10**(-10))*(2*corXY + 10**(-2)))/((miuX**2+miuY**2+10**(-10))*(stX**2+stY**2+10**(-2))) #not coefficient of correlation but covariance
        #         SSIMs += [SSIMtest]
        # SSIMs2 = pd.Series(SSIMs)
        # print(np.nanmean(SSIMs))
        # tableGSSI.loc[interval,seed]=np.nanmean(SSIMs)
        # # ___________________________linear regression _______________________________
        # from sklearn.linear_model import LinearRegression
        # model = LinearRegression( normalize=True, fit_intercept=False)
        # x= np.array(GTOD.values+.01).flatten().reshape(-1,1)
        # y = np.array(PLUOD.values+0.01).flatten()
        # model.fit(x, y)
        # tableSlope.loc[interval,seed] = model.coef_
        # tableRsq.loc[interval,seed] = model.score(x, y)
        # # ********************************* to be added ***************************

        SSIMs = []
        for i,gemrow in enumerate((amsterdamZones['gem_id'].append(pd.Series(0))).unique()):
            for j, gemcol in enumerate((amsterdamZones['gem_id'].append(pd.Series(0))).unique()):
                PLUwindow = PLUOD_home.loc[gemrow][gemcol]
                miuX = PLUwindow.values.mean()
                stX = PLUwindow.values.std()
                normPLUwindow = (PLUwindow-miuX)/stX
                GTwindow = GTOD_home.loc[gemrow][gemcol]
                miuY = GTwindow.values.mean()
                stY = GTwindow.values.std()
                normGTwindow = (GTwindow-miuY)/stY
                miuX=0
                miuY=0
                stX=1
                stY=1
                corXY = pd.Series(normPLUwindow.values.flatten()).corr(pd.Series(normGTwindow.values.flatten()),method='pearson')
                # corXY=np.cov(pd.Series(normPLUwindow.values.flatten()),pd.Series(normGTwindow.values.flatten())).mean()
                # corXY = np.cov(pd.Series(normPLUwindow.values.flatten()),pd.Series(normGTwindow.values.flatten())).mean()/(stX*stY)
                SSIMtest = ((2*miuX*miuY+10**(-10))*(2*corXY + 10**(-2)))/((miuX**2+miuY**2+10**(-10))*(stX**2+stY**2+10**(-2))) #not coefficient of correlation but covariance
                SSIMs += [SSIMtest]
        SSIMs2 = pd.Series(SSIMs)
        print(np.nanmean(SSIMs))
        tableGSSI_home.loc[interval,seed]=np.nanmean(SSIMs)
        # ___________________________linear regression _______________________________
        from sklearn.linear_model import LinearRegression
        model = LinearRegression( normalize=True, fit_intercept=False)
        x= np.array(GTOD_home.values+.01).flatten().reshape(-1,1)
        y = np.array(PLUOD_home.values+0.01).flatten()
        model.fit(x, y)
        tableSlope_home.loc[interval,seed] = model.coef_
        tableRsq_home.loc[interval,seed] = model.score(x, y)
        # print('The fitted model is nPLU = {slope}nGT + ({inter

        SSIMs = []
        for i, gemrow in enumerate((amsterdamZones['gem_id'].append(pd.Series(0))).unique()):
            for j, gemcol in enumerate((amsterdamZones['gem_id'].append(pd.Series(0))).unique()):
                PLUwindow = PLUOD_work.loc[gemrow][gemcol]
                miuX = PLUwindow.values.mean()
                stX = PLUwindow.values.std()
                normPLUwindow = (PLUwindow - miuX) / stX
                GTwindow = GTOD_work.loc[gemrow][gemcol]
                miuY = GTwindow.values.mean()
                stY = GTwindow.values.std()
                normGTwindow = (GTwindow - miuY) / stY
                miuX = 0
                miuY = 0
                stX = 1
                stY = 1
                corXY = pd.Series(normPLUwindow.values.flatten()).corr(pd.Series(normGTwindow.values.flatten()),
                                                                       method='pearson')
                # corXY=np.cov(pd.Series(normPLUwindow.values.flatten()),pd.Series(normGTwindow.values.flatten())).mean()
                # corXY = np.cov(pd.Series(normPLUwindow.values.flatten()),pd.Series(normGTwindow.values.flatten())).mean()/(stX*stY)
                SSIMtest = ((2 * miuX * miuY + 10 ** (-10)) * (2 * corXY + 10 ** (-2))) / (
                            (miuX ** 2 + miuY ** 2 + 10 ** (-10)) * (
                                stX ** 2 + stY ** 2 + 10 ** (-2)))  # not coefficient of correlation but covariance
                SSIMs += [SSIMtest]
        SSIMs2 = pd.Series(SSIMs)
        print(np.nanmean(SSIMs))
        tableGSSI_work.loc[interval, seed] = np.nanmean(SSIMs)
        # ___________________________linear regression _______________________________
        from sklearn.linear_model import LinearRegression
        model = LinearRegression( normalize=True, fit_intercept=False)
        x= np.array(GTOD_work.values+.01).flatten().reshape(-1,1)
        y = np.array(PLUOD_work.values+0.01).flatten()
        model.fit(x, y)
        tableSlope_work.loc[interval,seed] = model.coef_
        tableRsq_work.loc[interval,seed] = model.score(x, y)
        # print('The fitted model is nPLU = {slope}nGT + ({inter

        SSIMs = []
        for i, gemrow in enumerate((amsterdamZones['gem_id'].append(pd.Series(0))).unique()):
            for j, gemcol in enumerate((amsterdamZones['gem_id'].append(pd.Series(0))).unique()):
                PLUwindow = PLUOD_other.loc[gemrow][gemcol]
                miuX = PLUwindow.values.mean()
                stX = PLUwindow.values.std()
                normPLUwindow = (PLUwindow - miuX) / stX
                GTwindow = GTOD_other.loc[gemrow][gemcol]
                miuY = GTwindow.values.mean()
                stY = GTwindow.values.std()
                normGTwindow = (GTwindow - miuY) / stY
                miuX = 0
                miuY = 0
                stX = 1
                stY = 1
                corXY = pd.Series(normPLUwindow.values.flatten()).corr(pd.Series(normGTwindow.values.flatten()),
                                                                       method='pearson')
                # corXY=np.cov(pd.Series(normPLUwindow.values.flatten()),pd.Series(normGTwindow.values.flatten())).mean()
                # corXY = np.cov(pd.Series(normPLUwindow.values.flatten()),pd.Series(normGTwindow.values.flatten())).mean()/(stX*stY)
                SSIMtest = ((2 * miuX * miuY + 10 ** (-10)) * (2 * corXY + 10 ** (-2))) / (
                            (miuX ** 2 + miuY ** 2 + 10 ** (-10)) * (
                                stX ** 2 + stY ** 2 + 10 ** (-2)))  # not coefficient of correlation but covariance
                SSIMs += [SSIMtest]
        SSIMs2 = pd.Series(SSIMs)
        print(np.nanmean(SSIMs))
        tableGSSI_other.loc[interval, seed] = np.nanmean(SSIMs)
        # ___________________________linear regression _______________________________
        from sklearn.linear_model import LinearRegression
        model = LinearRegression( normalize=True, fit_intercept=False)
        x= np.array(GTOD_other.values+.01).flatten().reshape(-1,1)
        y = np.array(PLUOD_other.values+0.01).flatten()
        model.fit(x, y)
        tableSlope_other.loc[interval,seed] = model.coef_
        tableRsq_other.loc[interval,seed] = model.score(x, y)
        # print('The fitted model is nPLU = {slope}nGT + ({intercept})'.format(slope = model.coef_ , intercept = model.intercept_))
# # ********************* to be added ******************************
# tableGSSI.to_excel('D:/ax/gis/GSSItable_totalOD.xlsx',header=True,index=True)
# tableSum.to_excel('D:/ax/gis/sumtable_totalOD.xlsx',header=True,index=True)
# tableSlope.to_excel('D:/ax/gis/slopetable_totalOD.xlsx',header=True,index=True)
# tableRsq.to_excel('D:/ax/gis/Rsqtable_totalOD.xlsx',header=True,index=True)
# # *******************to be added *************************************

tableGSSI_home.to_excel('D:/ax/gis/GSSItable_home.xlsx',header=True,index=True)
tableSum_home.to_excel('D:/ax/gis/sumtable_home.xlsx',header=True,index=True)
tableSlope_home.to_excel('D:/ax/gis/slopetable_home.xlsx',header=True,index=True)
tableRsq_home.to_excel('D:/ax/gis/Rsqtable_home.xlsx',header=True,index=True)

tableGSSI_work.to_excel('D:/ax/gis/GSSItable_work.xlsx',header=True,index=True)
tableSum_work.to_excel('D:/ax/gis/sumtable_work.xlsx',header=True,index=True)
tableSlope_work.to_excel('D:/ax/gis/slopetable_work.xlsx',header=True,index=True)
tableRsq_work.to_excel('D:/ax/gis/Rsqtable_work.xlsx',header=True,index=True)

tableGSSI_other.to_excel('D:/ax/gis/GSSItable_other.xlsx',header=True,index=True)
tableSum_other.to_excel('D:/ax/gis/sumtable_other.xlsx',header=True,index=True)
tableSlope_other.to_excel('D:/ax/gis/slopetable_other.xlsx',header=True,index=True)
tableRsq_other.to_excel('D:/ax/gis/Rsqtable_other.xlsx',header=True,index=True)
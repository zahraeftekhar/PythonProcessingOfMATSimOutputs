import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
seeds=tuple(range(101,126))
intervals = tuple([30,60,300,600,900,1200,1500,1800,2100,2400,2700,3000,3300,3600,4500,5400,6300,7200])
amsterdamZones = pd.read_csv('D:/ax/gis/locationMappingToMezuroZones/CompleteAmsterdamMezuroZones.CSV')
# amsterdamZones = pd.read_csv('/data/zahraeftekhar/research_temporal/input_base/CompleteAmsterdamMezuroZones.CSV')

# ***************************** to be added ************************************
tableGSSI = pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)
tableSum = pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)
tableMAE = pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)
tableRsq = pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)
tableMSE = pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)
# #  ***************************** to be added ********************************

# tableGSSI_home = pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)
# tableSum_home = pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)
# tableSlope_home = pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)
# tableRsq_home = pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)
#
#
# tableGSSI_work = pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)
# tableSum_work = pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)
# tableSlope_work = pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)
# tableRsq_work = pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)
#
#
# tableGSSI_other = pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)
# tableSum_other = pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)
# tableSlope_other = pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)
# tableRsq_other= pd.DataFrame(np.zeros((len(intervals), len(seeds)), dtype=np.int32),columns=seeds,index=intervals)


arrays = [amsterdamZones['gem_id'].append(pd.Series(0)),amsterdamZones['mzr_id'].append(pd.Series(0))]
tuples = sorted(list(zip(*arrays)))
tuples.sort(key = lambda x: x[1])
index = pd.MultiIndex.from_tuples(tuples, names=['gem_id','mzr_id'])

# **********************to be added *************************
GTOD = pd.read_csv('D:/ax/gis/output_base/OD(06-30_09-30).CSV')
# GTOD = pd.read_csv('/data/zahraeftekhar/research_temporal/output_base/OD(06-30_09-30).CSV')
GTOD = GTOD.set_index(GTOD['Unnamed: 0'], drop=True)
GTOD = GTOD.iloc[:,1:]
GTOD = GTOD.sort_index(axis = 1)
GTOD = GTOD.sort_index(axis = 0)
GTOD = pd.DataFrame(GTOD.values, index=index, columns=index)
# # ************** to be added *******************************

# GTOD_home = pd.read_csv('D:/ax/OD(06-30_09-30)_home.CSV')
# GTOD_home = GTOD_home.set_index(GTOD_home['Unnamed: 0'], drop=True)
# GTOD_home = GTOD_home.iloc[:,1:]
# GTOD_home = GTOD_home.sort_index(axis = 1)
# GTOD_home = GTOD_home.sort_index(axis = 0)
# GTOD_home = pd.DataFrame(GTOD_home.values, index=index, columns=index)
# GTOD_work = pd.read_csv('D:/ax/OD(06-30_09-30)_work.CSV')
# GTOD_work = GTOD_work.set_index(GTOD_work['Unnamed: 0'], drop=True)
# GTOD_work = GTOD_work.iloc[:,1:]
# GTOD_work = GTOD_work.sort_index(axis = 1)
# GTOD_work = GTOD_work.sort_index(axis = 0)
# GTOD_work = pd.DataFrame(GTOD_work.values, index=index, columns=index)
# GTOD_other = pd.read_csv('D:/ax/OD(06-30_09-30)_other.CSV')
# GTOD_other = GTOD_other.set_index(GTOD_other['Unnamed: 0'], drop=True)
# GTOD_other = GTOD_other.iloc[:,1:]
# GTOD_other = GTOD_other.sort_index(axis = 1)
# GTOD_other = GTOD_other.sort_index(axis = 0)
# GTOD_other = pd.DataFrame(GTOD_other.values, index=index, columns=index)
for interval in intervals:
    print("************** interval is {inin} sec ***************".format(inin=interval))
    seeds = range(101,126)
    for seed in seeds:
        print("************** seed is {inin} ***************".format(inin=seed))
        # *******************to be added *****************************
        cellOD = pd.read_csv('D:/ax/gis/completePLUdata_{inter}sec/OD(06-30_09-30)_seed{ss}.CSV'.format(inter = interval,ss=seed)) #todo: please enter the address of the PLU OD
        # cellOD = pd.read_csv('/data/zahraeftekhar/research_temporal/completePLUdata_{inter}sec/OD(06-30_09-30)_seed{ss}.CSV'.format(inter = interval,ss=seed)) #todo: please enter the address of the PLU OD
        cellOD = cellOD.set_index(cellOD['Unnamed: 0'], drop=True)
        cellOD = cellOD.iloc[:,1:]
        cellOD2 = cellOD.sort_index(axis = 1)
        cellOD2 = cellOD2.sort_index(axis = 0)
        PLUOD = pd.DataFrame(cellOD2.values, index=index, columns=index)
        tableSum.loc[interval,seed] = np.sum(np.sum(PLUOD, axis=0))
        # *****************to be added ************************************

        # cellOD = pd.read_csv('D:/ax/gis/completePLUdata_{inter}sec/OD(06-30_09-30)_home_seed{ss}.CSV'.format(inter = interval,ss=seed)) #todo: please enter the address of the PLU OD
        # cellOD = cellOD.set_index(cellOD['Unnamed: 0'], drop=True)
        # cellOD = cellOD.iloc[:,1:]
        # cellOD2 = cellOD.sort_index(axis = 1)
        # cellOD2 = cellOD2.sort_index(axis = 0)
        # PLUOD_home = pd.DataFrame(cellOD2.values, index=index, columns=index)
        # tableSum_home.loc[interval,seed] = np.sum(np.sum(PLUOD_home, axis=0))
        #
        #
        # cellOD = pd.read_csv('D:/ax/gis/completePLUdata_{inter}sec/OD(06-30_09-30)_work_seed{ss}.CSV'.format(inter = interval,ss=seed)) #todo: please enter the address of the PLU OD
        # cellOD = cellOD.set_index(cellOD['Unnamed: 0'], drop=True)
        # cellOD = cellOD.iloc[:,1:]
        # cellOD2 = cellOD.sort_index(axis = 1)
        # cellOD2 = cellOD2.sort_index(axis = 0)
        # PLUOD_work = pd.DataFrame(cellOD2.values, index=index, columns=index)
        # tableSum_work.loc[interval,seed] = np.sum(np.sum(PLUOD_work, axis=0))
        #
        #
        # cellOD = pd.read_csv('D:/ax/gis/completePLUdata_{inter}sec/OD(06-30_09-30)_other_seed{ss}.CSV'.format(inter = interval,ss=seed)) #todo: please enter the address of the PLU OD
        # cellOD = cellOD.set_index(cellOD['Unnamed: 0'], drop=True)
        # cellOD = cellOD.iloc[:,1:]
        # cellOD2 = cellOD.sort_index(axis = 1)
        # cellOD2 = cellOD2.sort_index(axis = 0)
        # PLUOD_other = pd.DataFrame(cellOD2.values, index=index, columns=index)
        # tableSum_other.loc[interval,seed] = np.sum(np.sum(PLUOD_other, axis=0))
        # #
        # ****************************to be added ****************************
        SSIMs = []
        for i,gemrow in enumerate((amsterdamZones['gem_id'].append(pd.Series(0))).unique()):
            for j, gemcol in enumerate((amsterdamZones['gem_id'].append(pd.Series(0))).unique()):
                PLUwindow = PLUOD.loc[gemrow][gemcol]
                miuX = PLUwindow.values.mean()
                stX = PLUwindow.values.std()
                normPLUwindow = (PLUwindow-miuX)/stX
                GTwindow = GTOD.loc[gemrow][gemcol]
                miuY = GTwindow.values.mean()
                stY = GTwindow.values.std()
                normGTwindow = (GTwindow-miuY)/stY
                miuX=0
                miuY=0
                stX=1
                stY=1
                corXY = pd.Series(normPLUwindow.values.flatten()).corr(pd.Series(normGTwindow.values.flatten()),method='pearson')
                # covmat=np.vstack((normPLUwindow.values.flatten(), normGTwindow.values.flatten())).T
                # corXY=sum(covmat[:,0]*covmat[:,1])/len(covmat) %works same as the other way
                SSIMtest = ((2*miuX*miuY+10**(-10))*(2*corXY + 10**(-2)))/((miuX**2+miuY**2+10**(-10))*(stX**2+stY**2+10**(-2))) #not coefficient of correlation but covariance
                SSIMs += [SSIMtest]
        SSIMs2 = pd.Series(SSIMs)
        print(np.nanmean(SSIMs))
        tableGSSI.loc[interval,seed]=np.nanmean(SSIMs)
        # # ___________________________linear regression _______________________________
        # from sklearn.linear_model import LinearRegression
        # model = LinearRegression( normalize=True, fit_intercept=False)
        # x= np.array(GTOD.values+.01).flatten().reshape(-1,1)
        # y = np.array(PLUOD.values+0.01).flatten()
        # model.fit(x, y)
        # tableSlope.loc[interval,seed] = model.coef_
        # tableRsq.loc[interval,seed] = model.score(x, y)
        # ___________________________MSE _________________________________________
        from sklearn.metrics import mean_squared_error
        y_true= np.array(GTOD.values).flatten().reshape(-1,1)
        y_pred = np.array(PLUOD.values).flatten()
        tableMSE.loc[interval,seed]=mean_squared_error(y_true, y_pred)
        # ___________________________MAE _________________________________________
        from sklearn.metrics import mean_absolute_error
        y_true= np.array(GTOD.values).flatten().reshape(-1,1)
        y_pred = np.array(PLUOD.values).flatten()
        tableMAE.loc[interval,seed]=mean_absolute_error(y_true, y_pred)
        # ********************************* to be added ***************************

        # SSIMs = []
        # for i,gemrow in enumerate((amsterdamZones['gem_id'].append(pd.Series(0))).unique()):
        #     for j, gemcol in enumerate((amsterdamZones['gem_id'].append(pd.Series(0))).unique()):
        #         PLUwindow = PLUOD_home.loc[gemrow][gemcol]
        #         miuX = PLUwindow.values.mean()
        #         stX = PLUwindow.values.std()
        #         normPLUwindow = (PLUwindow-miuX)/stX
        #         GTwindow = GTOD_home.loc[gemrow][gemcol]
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
        # tableGSSI_home.loc[interval,seed]=np.nanmean(SSIMs)
        # # ___________________________linear regression _______________________________
        # from sklearn.linear_model import LinearRegression
        # model = LinearRegression( normalize=True, fit_intercept=False)
        # x= np.array(GTOD_home.values+.01).flatten().reshape(-1,1)
        # y = np.array(PLUOD_home.values+0.01).flatten()
        # model.fit(x, y)
        # tableSlope_home.loc[interval,seed] = model.coef_
        # tableRsq_home.loc[interval,seed] = model.score(x, y)
        # # print('The fitted model is nPLU = {slope}nGT + ({inter
        #
        # SSIMs = []
        # for i, gemrow in enumerate((amsterdamZones['gem_id'].append(pd.Series(0))).unique()):
        #     for j, gemcol in enumerate((amsterdamZones['gem_id'].append(pd.Series(0))).unique()):
        #         PLUwindow = PLUOD_work.loc[gemrow][gemcol]
        #         miuX = PLUwindow.values.mean()
        #         stX = PLUwindow.values.std()
        #         normPLUwindow = (PLUwindow - miuX) / stX
        #         GTwindow = GTOD_work.loc[gemrow][gemcol]
        #         miuY = GTwindow.values.mean()
        #         stY = GTwindow.values.std()
        #         normGTwindow = (GTwindow - miuY) / stY
        #         miuX = 0
        #         miuY = 0
        #         stX = 1
        #         stY = 1
        #         corXY = pd.Series(normPLUwindow.values.flatten()).corr(pd.Series(normGTwindow.values.flatten()),
        #                                                                method='pearson')
        #         # corXY=np.cov(pd.Series(normPLUwindow.values.flatten()),pd.Series(normGTwindow.values.flatten())).mean()
        #         # corXY = np.cov(pd.Series(normPLUwindow.values.flatten()),pd.Series(normGTwindow.values.flatten())).mean()/(stX*stY)
        #         SSIMtest = ((2 * miuX * miuY + 10 ** (-10)) * (2 * corXY + 10 ** (-2))) / (
        #                     (miuX ** 2 + miuY ** 2 + 10 ** (-10)) * (
        #                         stX ** 2 + stY ** 2 + 10 ** (-2)))  # not coefficient of correlation but covariance
        #         SSIMs += [SSIMtest]
        # SSIMs2 = pd.Series(SSIMs)
        # print(np.nanmean(SSIMs))
        # tableGSSI_work.loc[interval, seed] = np.nanmean(SSIMs)
        # # ___________________________linear regression _______________________________
        # from sklearn.linear_model import LinearRegression
        # model = LinearRegression( normalize=True, fit_intercept=False)
        # x= np.array(GTOD_work.values+.01).flatten().reshape(-1,1)
        # y = np.array(PLUOD_work.values+0.01).flatten()
        # model.fit(x, y)
        # tableSlope_work.loc[interval,seed] = model.coef_
        # tableRsq_work.loc[interval,seed] = model.score(x, y)
        # # print('The fitted model is nPLU = {slope}nGT + ({inter
        #
        # SSIMs = []
        # for i, gemrow in enumerate((amsterdamZones['gem_id'].append(pd.Series(0))).unique()):
        #     for j, gemcol in enumerate((amsterdamZones['gem_id'].append(pd.Series(0))).unique()):
        #         PLUwindow = PLUOD_other.loc[gemrow][gemcol]
        #         miuX = PLUwindow.values.mean()
        #         stX = PLUwindow.values.std()
        #         normPLUwindow = (PLUwindow - miuX) / stX
        #         GTwindow = GTOD_other.loc[gemrow][gemcol]
        #         miuY = GTwindow.values.mean()
        #         stY = GTwindow.values.std()
        #         normGTwindow = (GTwindow - miuY) / stY
        #         miuX = 0
        #         miuY = 0
        #         stX = 1
        #         stY = 1
        #         corXY = pd.Series(normPLUwindow.values.flatten()).corr(pd.Series(normGTwindow.values.flatten()),
        #                                                                method='pearson')
        #         # corXY=np.cov(pd.Series(normPLUwindow.values.flatten()),pd.Series(normGTwindow.values.flatten())).mean()
        #         # corXY = np.cov(pd.Series(normPLUwindow.values.flatten()),pd.Series(normGTwindow.values.flatten())).mean()/(stX*stY)
        #         SSIMtest = ((2 * miuX * miuY + 10 ** (-10)) * (2 * corXY + 10 ** (-2))) / (
        #                     (miuX ** 2 + miuY ** 2 + 10 ** (-10)) * (
        #                         stX ** 2 + stY ** 2 + 10 ** (-2)))  # not coefficient of correlation but covariance
        #         SSIMs += [SSIMtest]
        # SSIMs2 = pd.Series(SSIMs)
        # print(np.nanmean(SSIMs))
        # tableGSSI_other.loc[interval, seed] = np.nanmean(SSIMs)
        # # ___________________________linear regression _______________________________
        # from sklearn.linear_model import LinearRegression
        # model = LinearRegression( normalize=True, fit_intercept=False)
        # x= np.array(GTOD_other.values+.01).flatten().reshape(-1,1)
        # y = np.array(PLUOD_other.values+0.01).flatten()
        # model.fit(x, y)
        # tableSlope_other.loc[interval,seed] = model.coef_
        # tableRsq_other.loc[interval,seed] = model.score(x, y)
        # print('The fitted model is nPLU = {slope}nGT + ({intercept})'.format(slope = model.coef_ , intercept = model.intercept_))
#_____________________________________boxPlot of SSIMs _____________________________
# for interval in intervals:
#     plt.rcParams['figure.figsize'] = (16.0, 12.0)
#     plt.style.use('ggplot')
#     # y1 = tableGSSI.loc[interval,:]
#     # pdf1 = pd.Series(y1)
#     plt.figure(figsize=(6, 4.5))
#     fig1, ax1 = plt.subplots()
#     ax1 = plt.boxplot(tableGSSI,labels=(tableGSSI.index).astype(str))
#     # ax1.set_title("GSSI boxplot based on the polling interval")
#
#     plt.show()
#     # plt.xticks((np.arange(0, 24, step=4)),fontsize=11)
#     # plt.yticks((np.arange(0, 0.2, step=0.05)),fontsize=11)
#     # ax1.set_title(u'activity duration.',fontsize=16)
#     # ax1.set_xlabel(u'polling interval (sec)',fontsize=14)
#     ax1.set_ylabel('Frequency',fontsize=14)
#     plt.legend(['Gaussian Kernel density estimation','test set'],fontsize=11)
#     plt.savefig("D:/ax/gis/plots/GSSIboxplot.png",dpi = 300)
#     plt.savefig("D:/ax/gis/plots/GSSIboxplot.pdf",dpi = 300)
#_____________________________________boxPlot of SSIMs _____________________________
# for interval in intervals:
#     plt.rcParams['figure.figsize'] = (16.0, 12.0)
#     plt.style.use('ggplot')
#     # y1 = tableGSSI.loc[interval,:]
#     # pdf1 = pd.Series(y1)
#     plt.figure(figsize=(6, 4.5))
#     fig1, ax1 = plt.subplots()
#     ax1 = plt.boxplot(tableMSE,labels=(tableMSE.index).astype(str))
#     # ax1.set_title("GSSI boxplot based on the polling interval")
#
#     plt.show()
#     # plt.xticks((np.arange(0, 24, step=4)),fontsize=11)
#     # plt.yticks((np.arange(0, 0.2, step=0.05)),fontsize=11)
#     # ax1.set_title(u'activity duration.',fontsize=16)
#     # ax1.set_xlabel(u'polling interval (sec)',fontsize=14)
#     ax1.set_ylabel('Frequency',fontsize=14)
#     plt.legend(['Gaussian Kernel density estimation','test set'],fontsize=11)
#     plt.savefig("D:/ax/gis/plots/MSEboxplot.png",dpi = 300)
#     plt.savefig("D:/ax/gis/plots/MSEboxplot.pdf",dpi = 300)
# #
# ********************* to be added ******************************
tableGSSI.to_excel('D:/ax/gis/GSSItable_totalOD_final.xlsx',header=True,index=True)
tableSum.to_excel('D:/ax/gis/sumtable_totalOD_final.xlsx',header=True,index=True)
# tableSlope.to_excel('D:/ax/gis/slopetable_totalOD.xlsx',header=True,index=True)
# tableRsq.to_excel('D:/ax/gis/Rsqtable_totalOD.xlsx',header=True,index=True)
tableMSE.to_excel('D:/ax/gis/MSEtable_totalOD_final.xlsx',header=True,index=True)
tableMAE.to_excel('D:/ax/gis/MAEtable_totalOD_final.xlsx',header=True,index=True)

tableGSSI = pd.read_excel('D:/ax/gis/GSSItable_totalOD_final.xlsx')
tableGSSI.columns = ['seed',          101,          102,          103,          104,
                105,          106,          107,          108,          109,
                110,          111,          112,          113,          114,
                115,          116,          117,          118,          119,
                120,          121,          122,          123,          124,
                125]
# testint=pd.DataFrame()
tableGSSI = tableGSSI.set_index(['seed'],drop=True)
GSSIvalues = tableGSSI.values.flatten()
GSSIboxData = pd.DataFrame(GSSIvalues)
GSSIboxData['interval'] = 0
test = pd.concat([pd.Series([30,60,300,600,900,1200,1500,1800,2100,2400,2700,3000,3300,3600,4500,5400,6300,7200])] * 25, axis=1)

for i in range(0,18):
    GSSIboxData.loc[i*25:((i+1)*25-1),'interval']= np.transpose(test.loc[i,:].values)
GSSIboxData.columns = ['GSSI','interval']
GSSIboxData['interval']=GSSIboxData['interval']/60
import seaborn as sns
ax1 = sns.boxplot(y=GSSIboxData["interval"], x=GSSIboxData["GSSI"], linewidth=1,color='salmon',orient='h')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
ax1.set_xlabel('GSSI',fontsize=14)
ax1.set_ylabel('polling interval (min)',fontsize=14)
ax1.yaxis.grid(True)
# plt.xlim([.85,.92])
plt.savefig("D:/ax/gis/plots/GSSIboxplot_final.png", dpi=150)
plt.savefig("D:/ax/gis/plots/GSSIboxplot_final.pdf", dpi=150)


# reshape data*****
tableMAE = pd.read_excel('D:/ax/gis/MAEtable_totalOD_final.xlsx')
tableMAE.columns = ['seed',          101,          102,          103,          104,
                105,          106,          107,          108,          109,
                110,          111,          112,          113,          114,
                115,          116,          117,          118,          119,
                120,          121,          122,          123,          124,
                125]
# testint=pd.DataFrame()
tableMAE = tableMAE.set_index(['seed'],drop=True)
MAEvalues = tableMAE.values.flatten()
MAEboxData = pd.DataFrame(MAEvalues)
MAEboxData['interval'] = 0
test = pd.concat([pd.Series([30,60,300,600,900,1200,1500,1800,2100,2400,2700,3000,3300,3600,4500,5400,6300,7200])] * 25, axis=1)

for i in range(0,18):
    MAEboxData.loc[i*25:((i+1)*25-1),'interval']= np.transpose(test.loc[i,:].values)
MAEboxData.columns = ['MAE','interval']
MAEboxData['interval']=MAEboxData['interval']/60
import seaborn as sns
plt.figure()
ax1 = sns.boxplot(y=MAEboxData["interval"], x=MAEboxData["MAE"], linewidth=1,color='salmon',orient='h')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
ax1.set_xlabel('MAE',fontsize=14)
ax1.set_ylabel('polling interval (min)',fontsize=14)
ax1.yaxis.grid(True)
# plt.xlim([14,17.5])
plt.savefig("D:/ax/gis/plots/MAEboxplot_final.png", dpi=150)
plt.savefig("D:/ax/gis/plots/MAEboxplot_final.pdf", dpi=150)

# *******
# reshape data*****
# tableMSE = pd.read_excel('D:/ax/gis/MSEtable_totalOD_final.xlsx')
# tableMSE.columns = ['seed',          101,          102,          103,          104,
#                 105,          106,          107,          108,          109,
#                 110,          111,          112,          113,          114,
#                 115,          116,          117,          118,          119,
#                 120,          121,          122,          123,          124,
#                 125]
# # testint=pd.DataFrame()
# tableMSE = tableMSE.set_index(['seed'],drop=True)
# MSEvalues = tableMSE.values.flatten()
# MSEboxData = pd.DataFrame(MSEvalues)
# MSEboxData['interval'] = 0
# test = pd.concat([pd.Series([30,60,300,600,900,1200,1500,1800,2100,2400,2700,3000,3300,3600,4500,5400,6300,7200])] * 25, axis=1)
#
# for i in range(0,18):
#     MSEboxData.loc[i*25:((i+1)*25-1),'interval']= np.transpose(test.loc[i,:].values)
# MSEboxData.columns = ['MSE','interval']
# MSEboxData['interval']=MSEboxData['interval']/60
# import seaborn as sns
# plt.figure()
# ax1 = sns.boxplot(y=MSEboxData["interval"], x=MSEboxData["MSE"], linewidth=1,color='salmon',orient='h')
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# ax1.set_xlabel('MSE',fontsize=14)
# ax1.set_ylabel('polling interval (min)',fontsize=14)
# ax1.yaxis.grid(True)
# plt.xlim([14,17.5])
# plt.savefig("D:/ax/gis/plots/MSEboxplot_final.png", dpi=150)
# plt.savefig("D:/ax/gis/plots/MSEboxplot_final.pdf", dpi=150)
#
# # *******

# usersamplingLocActivity = pd.read_csv('D:/ax/gis/GTanalysis/userSampling_onePercentLocationActivityDetectionSensitivity.CSV')
# usersamplingLoc = pd.read_csv('D:/ax/gis/GTanalysis/userSampling_onePercentLocationDetectionSensitivity.CSV')
# usersamplingLoc.to_excel('D:/ax/gis/plots/kde_test/usersamplingLoc.xlsx',header=True)
# usersamplingLocActivity.to_excel('D:/ax/gis/plots/kde_test/usersamplingLocActivity.xlsx',header=True)
# *******



#
# tableGSSI.to_excel('/data/zahraeftekhar/research_temporal/GSSItable_totalOD.xlsx',header=True,index=True)
# tableSum.to_excel('/data/zahraeftekhar/research_temporal/sumtable_totalOD.xlsx',header=True,index=True)
# tableSlope.to_excel('/data/zahraeftekhar/research_temporal/slopetable_totalOD.xlsx',header=True,index=True)
# tableRsq.to_excel('/data/zahraeftekhar/research_temporal/Rsqtable_totalOD.xlsx',header=True,index=True)
# *******************to be added *************************************

# tableGSSI_home.to_excel('D:/ax/gis/GSSItable_home.xlsx',header=True,index=True)
# tableSum_home.to_excel('D:/ax/gis/sumtable_home.xlsx',header=True,index=True)
# tableSlope_home.to_excel('D:/ax/gis/slopetable_home.xlsx',header=True,index=True)
# tableRsq_home.to_excel('D:/ax/gis/Rsqtable_home.xlsx',header=True,index=True)
#
# tableGSSI_work.to_excel('D:/ax/gis/GSSItable_work.xlsx',header=True,index=True)
# tableSum_work.to_excel('D:/ax/gis/sumtable_work.xlsx',header=True,index=True)
# tableSlope_work.to_excel('D:/ax/gis/slopetable_work.xlsx',header=True,index=True)
# tableRsq_work.to_excel('D:/ax/gis/Rsqtable_work.xlsx',header=True,index=True)
#
# tableGSSI_other.to_excel('D:/ax/gis/GSSItable_other.xlsx',header=True,index=True)
# tableSum_other.to_excel('D:/ax/gis/sumtable_other.xlsx',header=True,index=True)
# tableSlope_other.to_excel('D:/ax/gis/slopetable_other.xlsx',header=True,index=True)
# tableRsq_other.to_excel('D:/ax/gis/Rsqtable_other.xlsx',header=True,index=True)
# ***************ranking the results based on their random seed to see
# which random seeds are most likely to produce outliers*******
# d = {'col1': [1, 2], 'col2': [3, 4]}
# df = pd.DataFrame(data=d)
# pd.DataFrame({col:str(col)+'=' for col in df}, index=df.index) + df.astype(str)
GSSIseedRanking = (round(tableGSSI*100000)/100000).astype(str)+','
GSSIseedRanking = GSSIseedRanking.astype(str) + pd.DataFrame({columns:str(columns) for columns in tableGSSI}, index=tableGSSI.index)



a = GSSIseedRanking.values
a.sort(axis=1)  # no ascending argument
a = a[:, ::-1]  # so reverse
a
GSSIseedRanking_new = pd.DataFrame(a, GSSIseedRanking.index)

# *************ranking for MSEs ***********
MSEseedRanking = (round(tableMSE*100000)/100000).astype(str)+','
MSEseedRanking = MSEseedRanking.astype(str) + pd.DataFrame({columns:str(columns) for columns in tableMSE}, index=tableMSE.index)



a = MSEseedRanking.values
a.sort(axis=1)  # no ascending argument
a = a[:, ::-1]  # so reverse
a
MSEseedRanking_new = pd.DataFrame(a, MSEseedRanking.index)
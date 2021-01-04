import pandas as pd
from scipy.stats import gaussian_kde
from shapely.geometry import Point
import time
from math import floor
import numpy as np
#_______________
intervals = [60,300,600,900,1200,1500,1800,2100,2400,2700,3000,3300,3600,4500,5400,6300,7200]
seed = 105
interval = 300
for interval in intervals:
    anchorLocs = pd.read_csv(
        'D:/ax/gis/anchors/anchorLocsPLU_{inter}sec_seed{ss}.CSV'.format(
            ss=seed, inter=interval), usecols=['VEHICLE', 'mzr_id', 'EASTING', 'NORTHING', 'start(sec)',
                                               'duration(sec)', 'activity'])
    print(min(anchorLocs['duration(sec)'])/60)
    anchorLocs['start'] = pd.to_timedelta(anchorLocs['start(sec)'],unit='s')
    anchorLocs['end'] = anchorLocs['start'] + pd.to_timedelta(anchorLocs['duration(sec)'],unit='s')
    anchorLocs['duration'] = pd.to_timedelta(anchorLocs['duration(sec)'],unit='s')
    datasample = anchorLocs.loc[0:500,['VEHICLE', 'mzr_id','start','duration','end', 'EASTING', 'NORTHING']]
    datasample['start'] = datasample['start'].astype(str)
    datasample['end'] = datasample['end'].astype(str)
    datasample['duration'] = datasample['duration'].astype(str)
    datasample['VEHICLE'] = (datasample['VEHICLE']).astype(int)
    datasample['mzr_id'] = (datasample['mzr_id']).astype(int)
    for i in range(0,501):
        datasample.loc[i,'start']=datasample.loc[i,'start'][0:3]+datasample.loc[i,'start'][7:15]
        datasample.loc[i,'end']=datasample.loc[i,'end'][0:3]+datasample.loc[i,'end'][7:15]
        datasample.loc[i,'duration']=datasample.loc[i,'duration'][0:3]+datasample.loc[i,'duration'][7:15]

    datasample.to_excel('D:/ax/gis/anchors/sample{inter}.xlsx'.format(inter=interval),index=False, header=True)
intervals = [1500]
for interval in intervals:
    anchorLocs = pd.read_csv(
        'D:/ax/gis/anchors/anchorLocs_fixedStart_{inter}sec_seed{ss}.CSV'.format(
            ss=seed, inter=interval), usecols=['VEHICLE', 'mzr_id', 'EASTING', 'NORTHING', 'start(sec)',
                                               'duration(sec)', 'activity'])
    print(min(anchorLocs['duration(sec)'])/60)
    anchorLocs['start'] = pd.to_timedelta(anchorLocs['start(sec)'],unit='s')
    anchorLocs['end'] = anchorLocs['start'] + pd.to_timedelta(anchorLocs['duration(sec)'],unit='s')
    anchorLocs['duration'] = pd.to_timedelta(anchorLocs['duration(sec)'],unit='s')
    datasample = anchorLocs.loc[0:500,['VEHICLE', 'mzr_id','start','duration','end', 'EASTING', 'NORTHING']]
    datasample['start'] = datasample['start'].astype(str)
    datasample['end'] = datasample['end'].astype(str)
    datasample['duration'] = datasample['duration'].astype(str)
    datasample['VEHICLE'] = (datasample['VEHICLE']).astype(int)
    datasample['mzr_id'] = (datasample['mzr_id']).astype(int)
    for i in range(0,501):
        datasample.loc[i,'start']=datasample.loc[i,'start'][0:3]+datasample.loc[i,'start'][7:15]
        datasample.loc[i,'end']=datasample.loc[i,'end'][0:3]+datasample.loc[i,'end'][7:15]
        datasample.loc[i,'duration']=datasample.loc[i,'duration'][0:3]+datasample.loc[i,'duration'][7:15]

    datasample.to_excel('D:/ax/gis/anchors/sample{inter}_fixed.xlsx'.format(inter=interval),index=False, header=True)


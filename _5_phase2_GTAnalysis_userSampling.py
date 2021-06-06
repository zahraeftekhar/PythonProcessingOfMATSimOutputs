import numpy as np
import pandas as pd
import time
import random
from scipy.stats import gaussian_kde
######################################### importing XML file plan ######################################################
import pickle
with open('D:/ax/gis/phase2/activities_df.pickle', 'rb') as handle:
    activities = pickle.load(handle)
with open('D:/ax/gis/phase2/trips_df.pickle', 'rb') as handle:
    trips = pickle.load(handle)
locProb = pd.read_csv("D:/ax/gis/phase2/locProbability.csv")
locProb["mzr_id"] = (locProb["mzr_id"]).astype(str)
gisInitialOutput = pd.read_csv("D:/ax/gis/phase2/gisOutput_activitiesLoc.CSV",usecols=["id","mzr_id","start"],
                               dtype=(str,str))
gisInitialOutput["start"] = pd.to_timedelta(gisInitialOutput["start"])
activities = gisInitialOutput.merge(activities, on=["id","start"], how="outer", sort=True)
activities["mzr_id"][activities["mzr_id"].isnull()] = "0"
activities = locProb.merge(activities, on=["mzr_id"], how="right", sort=True)
activities["home"][activities["home"].isnull()] = "0.33"
activities["work"][activities["work"].isnull()] = "0.33"
activities["other"][activities["other"].isnull()] = "0.33"
del activities["deviance"]
######################################### deriving activity duration and traveling duration from plan files ###########################################
start_time = time.time()
# _______________test seed based on user sampling _________________________________
seedSet = np.arange(101,151)
sensitivityTable = pd.DataFrame(index=seedSet)
sensitivityTable.index=seedSet
seed = 101
s=0
for s, seed in enumerate(seedSet):
    print(s, end='______ time:{tt}sec\n'.format(tt=time.time() - start_time))
    ids = pd.unique(activities["id"])
    random.seed(seed)
    indices = random.sample(range(len(ids)),round(.01*len(ids))) #1% sampling of users
    indices = pd.DataFrame(indices, columns=["id"])
    trainIDs =pd.DataFrame(ids[indices])
    trainIDs.columns = ["id"]
    trainingSet_trips = (trainIDs.merge(trips, on=["id"], how="inner", sort=True,validate="1:m"))
    trainingSet_activities = (trainIDs.merge(activities, on=["id"], how="inner", sort=True,validate="1:m"))
    trainingSet_home = trainingSet_activities[trainingSet_activities["type"] == "home"]
    trainingSet_work = trainingSet_activities[trainingSet_activities["type"] == "work"]
    trainingSet_other = trainingSet_activities[trainingSet_activities["type"] == "other"]

    # ________________________________________________________________________________________
    # ****************************************************************************************
    home = activities[activities["type"] == "home"]
    work = activities[activities["type"] == "work"]
    other = activities[activities["type"] == "other"]

    # ****************************************************************************************
    # ________________________ writing data: Trainset _________________________

    # ________________________________________________________________

    # ___________________________ Probability calculations_________________________________________
    #***************************************************************************
    prior_activity = len(activities)/(len(activities) + len(trips))
    prior_trip =  len(trips)/(len(activities) + len(trips))
    activities['Log prob activity'] = np.log(prior_activity)+np.log(
        gaussian_kde(trainingSet_activities['duration'].dt.total_seconds())
            .pdf(activities['duration'].dt.total_seconds()))+np.log(
        gaussian_kde(trainingSet_activities['start'].dt.total_seconds())
            .pdf(activities['start'].dt.total_seconds()))

    activities['Log prob trip'] = np.log(prior_trip)+np.log(
        gaussian_kde(trainingSet_trips['duration'].dt.total_seconds())
            .pdf(activities['duration'].dt.total_seconds()))+np.log(
        gaussian_kde(trainingSet_trips['start'].dt.total_seconds())
            .pdf(activities['start'].dt.total_seconds()))
    activities['activity?'] = activities['Log prob activity']>activities['Log prob trip']
    sensitivityTable.loc[seed,"stay accuracy"] = sum(activities['activity?'])/len(activities['activity?'])
    # print(sum(activities['activity?'])/len(activities['activity?']))
    #***************************************************************************
    trips['Log prob activity'] = np.log(prior_activity)+np.log(
        gaussian_kde(trainingSet_activities['duration'].dt.total_seconds())
            .pdf(trips['duration'].dt.total_seconds()))+np.log(
        gaussian_kde(trainingSet_activities['start'].dt.total_seconds())
            .pdf(trips['start'].dt.total_seconds()))

    trips['Log prob trip'] = np.log(prior_trip)+np.log(
        gaussian_kde(trainingSet_trips['duration'].dt.total_seconds())
            .pdf(trips['duration'].dt.total_seconds()))+np.log(
        gaussian_kde(trainingSet_trips['start'].dt.total_seconds())
            .pdf(trips['start'].dt.total_seconds()))
    trips['trip?'] = trips['Log prob activity']<trips['Log prob trip']
    sensitivityTable.loc[seed,"pass-by accuracy"] = sum(trips['trip?'])/len(trips['trip?'])

    # print(sum(trips['trip?'])/len(trips['trip?']))
    #***************************************************************************

    prior_home = len(trainingSet_home)/(len(trainingSet_home) + len(trainingSet_work) + len(trainingSet_other))
    prior_work =  len(trainingSet_work)/(len(trainingSet_home) + len(trainingSet_work) + len(trainingSet_other))
    prior_other = len(trainingSet_other)/(len(trainingSet_home) + len(trainingSet_work) + len(trainingSet_other))

    home['Log prob home'] = np.log(prior_home)+np.log(
        gaussian_kde(trainingSet_home['duration'].dt.total_seconds())
            .pdf(home['duration'].dt.total_seconds()))+np.log(
        gaussian_kde(trainingSet_home['start'].dt.total_seconds())
            .pdf(home['start'].dt.total_seconds()))+np.log((home["home"]).astype(float)) #todo: problem of log(0) for loc

    home['Log prob work'] = np.log(prior_work) + np.log(
        gaussian_kde(trainingSet_work['duration'].dt.total_seconds())
            .pdf(home['duration'].dt.total_seconds())) + np.log(
        gaussian_kde(trainingSet_work['start'].dt.total_seconds())
            .pdf(home['start'].dt.total_seconds())) + np.log(
        (home["work"]).astype(float))  # todo: problem of log(0) for loc

    home['Log prob other'] = np.log(prior_other) + np.log(
        gaussian_kde(trainingSet_other['duration'].dt.total_seconds())
            .pdf(home['duration'].dt.total_seconds())) + np.log(
        gaussian_kde(trainingSet_other['start'].dt.total_seconds())
            .pdf(home['start'].dt.total_seconds())) + np.log(
        (home["other"]).astype(float))  # todo: problem of log(0) for loc

    home['activity?'] = home[['Log prob home','Log prob work','Log prob other']].idxmax(axis=1)
    sensitivityTable.loc[seed,"home accuracy"] = len(home[home['activity?']=='Log prob home'])/len(home)

    # print("home accuracy: ",len(home[home['activity?']=='Log prob home'])/len(home))
#***************************************************************************
    work['Log prob home'] = np.log(prior_home)+np.log(
        gaussian_kde(trainingSet_home['duration'].dt.total_seconds())
            .pdf(work['duration'].dt.total_seconds()))+np.log(
        gaussian_kde(trainingSet_home['start'].dt.total_seconds())
            .pdf(work['start'].dt.total_seconds()))+np.log((work["home"]).astype(float)) #todo: problem of log(0) for


    work['Log prob work'] = np.log(prior_work) + np.log(
        gaussian_kde(trainingSet_work['duration'].dt.total_seconds())
            .pdf(work['duration'].dt.total_seconds())) + np.log(
        gaussian_kde(trainingSet_work['start'].dt.total_seconds())
            .pdf(work['start'].dt.total_seconds())) + np.log(
        (work["work"]).astype(float))  # todo: problem of log(0) for loc

    work['Log prob other'] = np.log(prior_other) + np.log(
        gaussian_kde(trainingSet_other['duration'].dt.total_seconds())
            .pdf(work['duration'].dt.total_seconds())) + np.log(
        gaussian_kde(trainingSet_other['start'].dt.total_seconds())
            .pdf(work['start'].dt.total_seconds())) + np.log(
        (work["other"]).astype(float))  # todo: problem of log(0) for loc

    work['activity?'] = work[['Log prob home','Log prob work','Log prob other']].idxmax(axis=1)
    sensitivityTable.loc[seed, "work accuracy"] = len(work[work['activity?']=='Log prob work'])/len(work)
    # print("work accuracy: ",len(work[work['activity?']=='Log prob work'])/len(work))
#***************************************************************************
    other['Log prob home'] = np.log(prior_home)+np.log(
        gaussian_kde(trainingSet_home['duration'].dt.total_seconds())
            .pdf(other['duration'].dt.total_seconds()))+np.log(
        gaussian_kde(trainingSet_home['start'].dt.total_seconds())
            .pdf(other['start'].dt.total_seconds()))+np.log((other["home"]).astype(float)) #todo: problem of log(0) for
    # loc

    other['Log prob work'] = np.log(prior_work) + np.log(
        gaussian_kde(trainingSet_work['duration'].dt.total_seconds())
            .pdf(other['duration'].dt.total_seconds())) + np.log(
        gaussian_kde(trainingSet_work['start'].dt.total_seconds())
            .pdf(other['start'].dt.total_seconds())) + np.log(
        (other["work"]).astype(float))  # todo: problem of log(0) for loc

    other['Log prob other'] = np.log(prior_other) + np.log(
        gaussian_kde(trainingSet_other['duration'].dt.total_seconds())
            .pdf(other['duration'].dt.total_seconds())) + np.log(
        gaussian_kde(trainingSet_other['start'].dt.total_seconds())
            .pdf(other['start'].dt.total_seconds())) + np.log(
        (other["other"]).astype(float))  # todo: problem of log(0) for loc

    other['activity?'] = other[['Log prob home','Log prob work','Log prob other']].idxmax(axis=1)
    sensitivityTable.loc[seed, "other accuracy"] = len(other[other['activity?']=='Log prob other'])/len(other)
#   print("other accuracy: ",len(other[other['activity?']=='Log prob other'])/len(other))
#     ***************************************************************************

sensitivityTable.to_excel("D:/ax/gis/phase2/userSampling_onePercentLocationDetectionSensitivity.xlsx", header=True,
                          index=True)
# sensitivityTable.to_csv("/data/zahraeftekhar/research_temporal/GTanalysis/userSampling_onePercentLocationDetectionSensitivity.CSV", header=True,index=False)

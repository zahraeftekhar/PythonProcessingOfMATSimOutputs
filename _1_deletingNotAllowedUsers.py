import xml.etree.ElementTree as ET
import time
import pandas as pd
startTime = time.time()
tree = ET.parse("D:/ax/gis/input_base/1.experienced_plans.xml")
# tree = ET.parse("/data/zahraeftekhar/research_temporal/input_base/1.experienced_plans.xml")
root = tree.getroot()
RemoveIDs = [(int)]
for child in root.findall('person'):
    aa = len(list(child.iter('leg')))
    ID2 = [int(child.get('id'))]
    if child.findall('plan/leg[@mode="car"]').__len__() != aa:
        RemoveIDs += [ID2]
        root.remove(child)
XMLFileName = "D:/ax/gis/output_base/PlanWithOnlyCar_again.xml"
# XMLFileName = "/data/zahraeftekhar/research_temporal/output_base/PlanWithOnlyCar_again.xml"
tree.write(str(XMLFileName), encoding="UTF-8", method="xml",
           xml_declaration=True)
forbiddenIDs = []
noGeneric = []
for m, person in enumerate(root.findall('person')):
    legListExperienced = person.findall('plan/leg/route')
    nn=0
    for n in range(len(legListExperienced)):
        if legListExperienced[n].get('type') == 'generic':
            nn+=1
    if nn==0 and person.findall('plan/activity')[0].get('type') == \
            person.findall('plan/activity')[-1].get('type'):
        noGeneric+=[person.get('id')]
    else:root.remove(person)
XMLFileName = "D:/ax/gis/output_base/PlanWithOnlyCar_again_NoGeneric.xml"
# XMLFileName = "/data/zahraeftekhar/research_temporal/output_base/PlanWithOnlyCar_again_NoGeneric.xml"
tree.write(str(XMLFileName), encoding="UTF-8", method="xml",
           xml_declaration=True)
# ____________ deleting users with zero duration activities __________________________
negDurationIDs=[]
for m, person in enumerate(root.findall('person')):
    activityList = person.findall('plan/activity')
    for n in range(1,len(activityList)-1):
        if pd.to_timedelta(activityList[n].get('end_time')).total_seconds() - pd.to_timedelta(
                activityList[n].get('start_time')).total_seconds() <= 0:
            negDurationIDs+=[person.get("id")]
            root.remove(person)
            break
negDurationIDs2=[]
for m, person in enumerate(root.findall('person')):
    activityList = person.findall('plan/activity')
    for n in range(1,len(activityList)-1):
        if pd.to_timedelta(activityList[n].get('end_time')).total_seconds() - pd.to_timedelta(
                activityList[n].get('start_time')).total_seconds() <= 0:
            negDurationIDs2+=[person.get("id")]
            root.remove(person)
            break

XMLFileName = "D:/ax/gis/output_base/PlanWithOnlyCar_again_NoGeneric_NoZeroDuationActivity_testttt.xml"
# XMLFileName = "/data/zahraeftekhar/research_temporal/output_base/PlanWithOnlyCar_again_NoGeneric_NoZeroDuationActivity.xml"
tree.write(str(XMLFileName), encoding="UTF-8", method="xml",
           xml_declaration=True)

#  ___________________________________________________________________________________

ids=[]
for person in root.findall('person'):
    ids+=[person.get("id")]

print(time.time() - startTime)  # 70 seconds
# ___________________________deleting from snapshot file _______________
# snapFile = pd.read_csv("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot_final/ITERS/it.1/snapShot.CSV",delimiter="\t")
snapFile = pd.read_csv("/data/zahraeftekhar/research_temporal/input_base/snapShot.CSV",delimiter="\t")
snapFile = snapFile.sort_values(by=["VEHICLE"])
kkk= pd.DataFrame()
startTime=time.time()
i=0
mmm=1
snapModified=pd.DataFrame()
for i in range(0,3000): # range(len(ids))
    if i == mmm * 100:
        print('{percentage} percent____{duration} sec'.format(percentage=i / len(ids),
                                                              duration=time.time() - startTime))
        mmm += 1
    kk=snapFile[snapFile['VEHICLE']==int(ids[i])]
    snapModified=snapModified.append(kk)
kkk = kkk.append(snapModified)
print("(0,3000) is finished")
snapModified=pd.DataFrame()
for i in range(3000,6000): # range(len(ids))
    if i == mmm * 100:
        print('{percentage} percent____{duration} sec'.format(percentage=i / len(ids),
                                                              duration=time.time() - startTime))
        mmm += 1
    kk=snapFile[snapFile['VEHICLE']==int(ids[i])]
    snapModified=snapModified.append(kk)
kkk = kkk.append(snapModified)
print("(3000,6000) is finished")
snapModified=pd.DataFrame()
for i in range(6000,9000): # range(len(ids))
    if i == mmm * 100:
        print('{percentage} percent____{duration} sec'.format(percentage=i / len(ids),
                                                              duration=time.time() - startTime))
        mmm += 1
    kk=snapFile[snapFile['VEHICLE']==int(ids[i])]
    snapModified=snapModified.append(kk)
kkk = kkk.append(snapModified)
print("(6000,9000) is finished")
snapModified=pd.DataFrame()
for i in range(9000,12000): # range(len(ids))
    if i == mmm * 100:
        print('{percentage} percent____{duration} sec'.format(percentage=i / len(ids),
                                                              duration=time.time() - startTime))
        mmm += 1
    kk=snapFile[snapFile['VEHICLE']==int(ids[i])]
    snapModified=snapModified.append(kk)
kkk = kkk.append(snapModified)
print("(9000,12000) is finished")
snapModified=pd.DataFrame()
for i in range(12000,15000): # range(len(ids))
    if i == mmm * 100:
        print('{percentage} percent____{duration} sec'.format(percentage=i / len(ids),
                                                              duration=time.time() - startTime))
        mmm += 1
    kk=snapFile[snapFile['VEHICLE']==int(ids[i])]
    snapModified=snapModified.append(kk)
kkk = kkk.append(snapModified)
print("(12000,15000) is finished")
snapModified=pd.DataFrame()
for i in range(15000,18000): # range(len(ids))
    if i == mmm * 100:
        print('{percentage} percent____{duration} sec'.format(percentage=i / len(ids),
                                                              duration=time.time() - startTime))
        mmm += 1
    kk=snapFile[snapFile['VEHICLE']==int(ids[i])]
    snapModified=snapModified.append(kk)
kkk = kkk.append(snapModified)
print("(15000,18000) is finished")
snapModified=pd.DataFrame()
for i in range(18000,21000): # range(len(ids))
    if i == mmm * 100:
        print('{percentage} percent____{duration} sec'.format(percentage=i / len(ids),
                                                              duration=time.time() - startTime))
        mmm += 1
    kk=snapFile[snapFile['VEHICLE']==int(ids[i])]
    snapModified=snapModified.append(kk)
kkk = kkk.append(snapModified)
print("(18000,21000) is finished")
snapModified=pd.DataFrame()
for i in range(21000,len(ids)): # range(len(ids))
    if i == mmm * 100:
        print('{percentage} percent____{duration} sec'.format(percentage=i / len(ids),
                                                              duration=time.time() - startTime))
        mmm += 1
    kk=snapFile[snapFile['VEHICLE']==int(ids[i])]
    snapModified=snapModified.append(kk)
kkk = kkk.append(snapModified)
# kkk.to_csv("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot_final/ITERS/it.1/snapShot_allowedUsers.CSV")
kkk.to_csv("/data/zahraeftekhar/research_temporal/output_base/snapShot_allowedUsers.CSV")
print(time.time() - startTime)
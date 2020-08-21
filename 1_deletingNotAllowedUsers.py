import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, tostring, Comment
from xml.dom import minidom
import xml
import lxml
from lxml.etree import xmlfile
from lxml import etree
import numpy as np
from typing import List
import time
import pandas as pd
startTime = time.time()
tree = ET.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/ITERS/it.1/1.experienced_plans.xml")
root = tree.getroot()
RemoveIDs = [(int)]
for child in root.findall('person'):
    aa = len(list(child.iter('leg')))
    ID2 = [int(child.get('id'))]
    if child.findall('plan/leg[@mode="car"]').__len__() != aa:
        RemoveIDs += [ID2]
        root.remove(child)

XMLFileName = "C:/Users/zahraeftekhar/eclipse-workspace/matsim_directory/matsim-example-project/scenarios/" \
              "example_zahra/Amsterdam/original files/PlanWithOnlyCar_again.xml"
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
XMLFileName = "C:/Users/zahraeftekhar/eclipse-workspace/matsim_directory/matsim-example-project/scenarios/" \
              "example_zahra/Amsterdam/original files/PlanWithOnlyCar_again_NoGeneric.xml"
tree.write(str(XMLFileName), encoding="UTF-8", method="xml",
           xml_declaration=True)
ids=[]
for person in root.findall('person'):
    ids+=[person.get("id")]

print(time.time() - startTime)  # 70 seconds
# ___________________________deleting from snapshot file _______________
snapFile = pd.read_csv("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/ITERS/it.1/snapShot.CSV",delimiter="\t")
snapFile = snapFile.sort_values(by=["VEHICLE"])
kkk= pd.DataFrame()
startTime=time.time()
i=0
mmm=1
snapModified=pd.DataFrame()
for i in range(0,5000): # range(len(ids))
    if i == mmm * 100:
        print('{percentage} percent____{duration} sec'.format(percentage=i / len(ids),
                                                              duration=time.time() - startTime))
        mmm += 1
    kk=snapFile[snapFile['VEHICLE']==int(ids[i])]
    snapModified=snapModified.append(kk)
kkk = kkk.append(kkk)
snapModified=pd.DataFrame()
for i in range(5000,10000): # range(len(ids))
    if i == mmm * 100:
        print('{percentage} percent____{duration} sec'.format(percentage=i / len(ids),
                                                              duration=time.time() - startTime))
        mmm += 1
    kk=snapFile[snapFile['VEHICLE']==int(ids[i])]
    snapModified=snapModified.append(kk)
kkk = kkk.append(kkk)
snapModified=pd.DataFrame()
for i in range(10000,15000): # range(len(ids))
    if i == mmm * 100:
        print('{percentage} percent____{duration} sec'.format(percentage=i / len(ids),
                                                              duration=time.time() - startTime))
        mmm += 1
    kk=snapFile[snapFile['VEHICLE']==int(ids[i])]
    snapModified=snapModified.append(kk)
kkk = kkk.append(kkk)
snapModified=pd.DataFrame()
for i in range(15000,20000): # range(len(ids))
    if i == mmm * 100:
        print('{percentage} percent____{duration} sec'.format(percentage=i / len(ids),
                                                              duration=time.time() - startTime))
        mmm += 1
    kk=snapFile[snapFile['VEHICLE']==int(ids[i])]
    snapModified=snapModified.append(kk)
kkk = kkk.append(kkk)
snapModified=pd.DataFrame()
for i in range(20000,len(ids)): # range(len(ids))
    if i == mmm * 100:
        print('{percentage} percent____{duration} sec'.format(percentage=i / len(ids),
                                                              duration=time.time() - startTime))
        mmm += 1
    kk=snapFile[snapFile['VEHICLE']==int(ids[i])]
    snapModified=snapModified.append(kk)
kkk = kkk.append(kkk)
kkk.to_csv("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/ITERS/it.1/snapShot_allowedUsers.CSV")

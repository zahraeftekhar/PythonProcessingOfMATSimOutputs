
# importing XML file plan
from xml.dom import minidom
xmlPlan = minidom.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.plans.xml")
itemlist = xmlPlan.getElementsByTagName('person')
del xmlPlan
# importing XML file experienced plan
from xml.dom import minidom
xmlExperienced = minidom.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.experienced_plans.xml")
itemlist = xmlExperienced.getElementsByTagName('person')
del xmlExperienced
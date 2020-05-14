
# importing XML file plan
from xml.dom import minidom
xmldoc = minidom.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.plans.xml")
itemlist = xmldoc.getElementsByTagName('person')
del xmldoc
# importing XML file experienced plan
from xml.dom import minidom
xmldoc = minidom.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.experienced_plans.xml")
itemlist = xmldoc.getElementsByTagName('person')
del xmldoc
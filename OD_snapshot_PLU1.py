import pandas as pd

snapData = pd.read_csv("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar"
                       "_30secSnapShot/ITERS/it.1/1snapshot.csv", sep="\t",
                       usecols=["VEHICLE", "TIME", "EASTING", "NORTHING"])
snapData = snapData.set_index("VEHICLE")
snapData = snapData.sort_index()
snapData = snapData.sort_values(by=["VEHICLE","TIME"], na_position='first')
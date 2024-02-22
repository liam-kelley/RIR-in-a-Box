"../gwa_3Dfront.csv  = room_name, Source_Pos, Receiver_Pos, T60_125Hz,T60_250Hz,T60_500Hz,T60_1000Hz,T60_2000Hz,T60_4000Hz,T60_8000Hz"
"../GWA_Dataset_small/stats.csv = Path,T60_125Hz,T60_250Hz,T60_500Hz,T60_1000Hz,T60_2000Hz,T60_4000Hz,T60_8000Hz,Source_Pos,Receiver_Pos"

import pandas as pd

# from GWA_Dataset_small/stats.csv import all columns into gwa_3Dfront.csv
# the Path column will be transformed into the room_name column.
#     This shall be done by removing the 13 last characters from the items in Path column
# the other columns will be copied as is

gwa_df = pd.read_csv("../GWA_Dataset_small/stats.csv")


# remove the last 13 characters from the Path column
gwa_df['mesh_name'] = gwa_df['Path'].apply(lambda x: x[:-13] + "/house.obj")

# rename the Path column to room_name
gwa_df.rename(columns={"Path": "rir_name"}, inplace=True)

# save the merged dataframe to a new csv file
gwa_df.to_csv("../gwa_3Dfront.csv", index=False)

# print the first 5 rows of the merged dataframe
print(gwa_df.head())

import pandas as pd
import glob

def get_df_list(df, meshes):
    # Create a new DataFrame for each row in df, repeat it for each mesh, and concatenate them all
    df_new_list = []
    for index, row in df.iterrows():
        temp_df = pd.DataFrame([row]*len(meshes), columns=df.columns)
        temp_df["mesh"] = meshes
        df_new_list.append(temp_df)
    return pd.concat(df_new_list).reset_index(drop=True)

# Load the dataset
df = pd.read_csv('datasets/ValidationDataset/subset/realval_dataset_og.csv')

print(df.head())

# drop some columns: SrcOrientatedTowards, SrcOrientedTowardsPointX, SrcOrientedTowardsPointY, SrcOrientedTowardsPointZ
df = df.drop(columns=['SrcOrientatedTowards', 'SrcOrientedTowardsPointX', 'SrcOrientedTowardsPointY', 'SrcOrientedTowardsPointZ'])
df = df.drop(columns=['RoomConfiguration'])
df = df.drop(columns=['SrcName'])

# Get just audio file name
df['audio_file_path'] = df['audio_file_path'].apply(lambda x: x.split('/')[-1])

# Get every line where WABR is 1
dfWABR = df[df['WABR'] == 1]
WABRs = glob.glob('datasets/ValidationDataset/fixed_meshes/WABR*obj')
WABRs = [x.split('/')[-1] for x in WABRs]
dfWABR = get_df_list(dfWABR, WABRs)

# Get every line where WACR is 1
dfWACR = df[df['WACR'] == 1]
WACRs = glob.glob('datasets/ValidationDataset/fixed_meshes/WACR*obj')
WACRs = [x.split('/')[-1] for x in WACRs]
dfWACR = get_df_list(dfWACR, WACRs)

# Get every line where LAS is 1
dfLAS = df[df['LAS'] == 1]
LASs = glob.glob('datasets/ValidationDataset/fixed_meshes/LAS*obj')
LASs = [x.split('/')[-1] for x in LASs]
dfLAS = get_df_list(dfLAS, LASs)

# Concatenate the DataFrames
df = pd.concat([dfWABR, dfWACR, dfLAS]).reset_index(drop=True)

# Save the dataset
df.to_csv('datasets/ValidationDataset/subset/realval_dataset.csv', index=False)
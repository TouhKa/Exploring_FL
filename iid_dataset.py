# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

DATA_DIR = "./data/"
OUT_DIR = "./out/"

file_app_usages = "App_usage_trace.txt"
Non_IID_file_name = "top_90_apps.csv"
IID_file_name = "top_90_apps_IID.csv"
q = 0.9

def read_traces():
  x =[]
  with open(DATA_DIR+file_app_usages) as fin:
      for idx, line in enumerate(fin):
          splitLine = line.rstrip().split()
          if idx != 0:
              splitLine = np.array([int(i) for i in splitLine])
              x.append(splitLine)
            
  x = np.array(x, dtype="int")
  return x
def main():
  data = read_traces()
  df = pd.DataFrame(data, columns= ["User_ID", "Timestamp", "BaseStation_ID", "App_ID", "TrafficInBytes"])
  print(df.describe())

  counted_apps = df.loc[:, ["App_ID"]].value_counts()
  print(f"10 most common apps:\n\n {counted_apps[:10]}\n")

  quantile = np.quantile(counted_apps.values, q)
  print(f"Quantile threshold: {quantile}")
  IID_used_apps = counted_apps.loc[counted_apps.values >= quantile, :]
  print(f"# Apps: {len(IID_used_apps)}")

  syntetic_ids = [i[0] for i in IID_used_apps.index]
  df_syntetic = df.loc[df["App_ID"].isin(syntetic_ids)].reset_index()
  #write non_iid_dataset to csv
  df_syntetic.to_csv(DATA_DIR+ Non_IID_file_name, encoding="utf8")

  #create IID dataset without users
  df_syntetic = df_syntetic.drop(["User_ID", "index"], axis=1)
  print(f"Discarded rows: {len(df)- len(df_syntetic)}")
  print(f"Remaining Rows: {len(df_syntetic)}")

  #every app has the same amount of entries
  min_usage = min(IID_used_apps)
  print(f"Min Usage: {min_usage}")

  #get random ids per app except
  random_idx = []
  for id in syntetic_ids[:-1]:
    idx = df_syntetic.loc[df_syntetic["App_ID"] == id].sample(frac = 1)[:min_usage].index
    for id in idx:
      random_idx.append(id)

  for id in df_syntetic.loc[df_syntetic["App_ID"] == syntetic_ids[-1]].index.values: 
    random_idx.append(id)
    
  random_idx.sort()

  df_syntetic_new = df_syntetic.iloc[random_idx]
  print(df_syntetic_new[:3])
  df_syntetic_new.to_csv(DATA_DIR+ IID_file_name, encoding="utf8")

if __name__ == "__main__":
  main()
import os
import json

import pandas as pd

csv_file = 'diffusiondb_100_ex.csv'

df = pd.read_csv(csv_file)


datas = []
for i,row in df.iterrows():
    prompt = row['text']
    datas.append({"iid":i,"prompt":prompt})

with open("diffusiondb_100_ex.jsonl", "w", encoding="utf-8") as f:
    for data in datas:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

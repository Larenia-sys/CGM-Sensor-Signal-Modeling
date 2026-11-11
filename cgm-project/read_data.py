import os
import pandas as pd

raw_path = "data/raw"
print("raw文件夹路径：", raw_path)
file_list = [f for f in os.listdir(raw_path) if f.endswith(".csv")]
print("找到的csv文件：", file_list)

if len(file_list) > 0:
    df = pd.read_csv(os.path.join(raw_path, file_list[0]))
    print("数据形状：", df.shape)
    print("列名：")
    print(df.columns.tolist())
    print("前5行：")
    print(df.head())
else:
    print("raw里面没有csv文件！")
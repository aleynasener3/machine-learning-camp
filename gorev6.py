import seaborn as sns

df = sns.load_dataset(("car_crashes"))
df.columns

#colum=["NUM_"+ col.upper() for col in df.columns if df[col].dtype!="O"]
colum=[col+"_flag" for col in df.columns if not "no" in col]

print(colum)
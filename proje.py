import pandas as pd
import numpy as np

#görev1

df=pd.read_csv(r'C:\Users\Aleyna\Desktop\csv\persona.csv')#soru1
df.nunique() #soru2
df["PRICE"].unique()#soru3
df["PRICE"].value_counts()#soru4
df["COUNTRY"].value_counts()#soru5
df.groupby("COUNTRY").agg({"PRICE":["sum"]})#soru6
df.groupby("SOURCE").agg({"PRICE":["count"]})#soru7
df.groupby("COUNTRY").agg({"PRICE":["mean"]})#soru8
df.groupby("SOURCE").agg({"PRICE":["sum"]})#soru9
df.groupby(["COUNTRY","SOURCE"]).agg({"PRICE":["sum"]})#soru10

#görev2
df=df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE":["mean"]})
#görev3
agg_df = df.groupby(by=["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)


#görev4
agg_df=agg_df.reset_index()

#görev5
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins = [0, 18, 23, 30, 40, 70],
                      labels = ["0_18", "19_23", "24_30", "31_40", "41_70"])
#görev6
agg_df['customers_level_based'] = agg_df[['COUNTRY', 'SOURCE', 'SEX', 'AGE_CAT']].agg(lambda x: '_'.join(x).upper(), axis=1)
agg_df = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"})
agg_df = agg_df.reset_index()

#görev7
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"],4,labels=["D","C","B","A"])
agg_df.groupby("SEGMENT").agg({"PRICE": "mean"})

#görev8
new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user]

new_user = "FRA_IOS_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user]
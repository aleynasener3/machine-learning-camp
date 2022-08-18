import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df=pd.read_csv(r'C:\Users\Aleyna\PycharmProjects\untitled2\datasets\diabetes.csv')
df.head()

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

def grab_col_names(dataframe, cat_th=5, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def target_analysis(dataframe):
    df_res=df.groupby("Outcome")[num_cols].mean()
    return df_res

def graphs(dataframe, col_name):
    if col_name in num_cols:
        dataframe[col_name].hist(bins=20)
        plt.xlabel(col_name)
        plt.title(col_name)
        plt.show()
    if col_name in cat_cols:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

graphs(df,"Outcome")

df_res=target_analysis(df)

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def outlier(dataframe,col_name):
    low,up=outlier_thresholds(dataframe,col_name)
    df2=dataframe.loc[((dataframe[col_name])<low) | (dataframe[col_name]>up),col_name]
    index=df2.index
    return df2,index

df2,index= outlier(df,"Insulin")
df2.value_counts()
missing_col_names=["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]

def missing_values(dataframe,col_name):
    df3=dataframe.loc[dataframe[col_name]==0,col_name]
    dataframe[col] = np.where(df[col] == 0, np.nan, dataframe[col])

    return df3

df["Glucose"].isnull().value_counts()
corr = df[num_cols].corr()

for col in missing_col_names:
    df3=missing_values(df, col)
    df[col] = df[col].fillna(df[col].mean())

def push(dataframe,col_name):
    low,up=outlier_thresholds(df, col_name)
    dataframe.loc[(df[col_name]<low), col_name]=low
    dataframe.loc[(df[col_name]>up),col_name]=up

for col in missing_col_names:
    push(df,col)

df["AGE_CAT"] = pd.cut(df["Age"], bins = [21, 41, 61, 81],
                      labels = ["21_41", "41_61", "61_81"])
df["bmi_cat"] = pd.qcut(df["BMI"],4,labels=["D","C","B","A"])

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df=one_hot_encoder(df,cat_cols)
df.head()

ss=StandardScaler()
df[num_cols]=ss.fit_transform(df[num_cols])
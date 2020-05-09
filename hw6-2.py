import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

house = pd.read_csv("train.csv")

plt.hist(house.SalePrice)
plt.title(" Ev ücretlerinin dağılımı",)
plt.xlabel("Ücretler")
plt.ylabel("Olay sayısı")
plt.show()


print(house.describe())
print(house.info())
print(house.nunique().head(50))
house['yeni_mi'] = np.where(house['YearBuilt']>=2005, 1, 0)

plt.figure(figsize=(14,6))

plt.subplot(1,3,1)
plt.bar(house.groupby("RoofStyle")["SalePrice"].mean().index,
        house.groupby("RoofStyle")["SalePrice"].mean())
plt.title("Çatı tipine göre ücretler")
plt.ylabel("Ortalama ücret")
plt.xlabel("Çatı tipi")
plt.xticks(np.arange(0,6))

plt.subplot(1,3,2)
plt.bar(house.groupby("BedroomAbvGr")["SalePrice"].mean().index,
        house.groupby("BedroomAbvGr")["SalePrice"].mean())
plt.title("Yatak odası sayısına göre ücretler")
plt.ylabel("Ortalama ücret")
plt.xlabel("Yatak odası sayısı")
plt.xticks(np.arange(0,9))

plt.subplot(1,3,3)
plt.bar(house.groupby("yeni_mi")["SalePrice"].mean().index,
        house.groupby("yeni_mi")["SalePrice"].mean())
plt.title("yıla göre ücretler")
plt.ylabel("Ortalama ücret")
plt.xlabel("0: 2005ten eski evler, 1: 2005ten yeni evler")
plt.xticks([0,1])

plt.tight_layout()
plt.show()


## regression işleri

Y = house["SalePrice"]
X = house[["BedroomAbvGr","yeni_mi"]]

lrm = linear_model.LinearRegression()
print(lrm.fit(X, Y))

print('Değişkenler: \n', lrm.coef_)
print('Sabit değer (bias): \n', lrm.intercept_)

X = sm.add_constant(X)
results = sm.OLS(Y, X).fit()
print(results.summary())

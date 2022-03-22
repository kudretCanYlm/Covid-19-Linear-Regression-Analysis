
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,median_absolute_error,r2_score

veriseti = pd.read_csv("1912709059_MKM411.csv",error_bad_lines=False)
X = veriseti["VAKA_SAYISI"].values
Y = veriseti["VEFAT_SAYISI"].values
print(X,Y)

X = X.reshape((len(X),1))

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=42)

model_regression = LinearRegression()
model_regression.fit(x_train,y_train)

y_pred = model_regression.predict(x_test)

###Test Seti###
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,model_regression.predict(x_train),color="blue")
plt.title("Brezilya Covid-19 İstatistikleri")
plt.xlabel("HAFTALAR")
plt.ylabel("VAKA_SAYISI ve VEFAT_SAYISI")
plt.show()

###Performans Değerlendirme###

print("R-Kare:",r2_score(y_test,y_pred))
print("MAE:",mean_absolute_error(y_test,y_pred))
print("MSE:",mean_squared_error(y_test,y_pred))
print("MedAE",median_absolute_error(y_test,y_pred))

###Regresyon Denklemi###

print("Eğim:",model_regression.coef_)
print("Kesen:",model_regression.intercept_)
print("y%0.2f"%model_regression.coef_+"x+%0.2f"%model_regression.intercept_)

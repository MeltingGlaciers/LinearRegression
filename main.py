import pandas as pd
import numpy as np
from matplotlib import pyplot as pt
from sklearn import linear_model as lm

file = pd.read_csv("D:\\Progs\\ISIS\Lab2\\beer2.csv")

data = []

for el in file.to_numpy():
    data.append(el[0])

data = data[1:]

months = np.array(list(range(1,len(data)+1)))

f = pt.figure()
f.set_figwidth(20)
f.set_figheight(10)

pt.plot(
     months,
     data,
     '-',
    color='b',)

model = lm.LinearRegression().fit(
    months.reshape(-1,1),
    data)

x_pred = np.array(range(1,len(data)+10)).reshape(-1,1)

y_pred = model.predict(x_pred)
print(y_pred[-8:])

pt.plot(x_pred,y_pred,'-',color='red',linewidth=5)

pt.xticks(list(range(1,len(data)+10)))
pt.show()
pt.clf()
f = pt.figure()
f.set_figwidth(20)
f.set_figheight(10)
pt.xticks(list(range(1,len(data)+10)))

anomaly = [data[i]-y_pred[i] for i in range(len(x_pred)-9)]

pt.plot([0,55],[0,0],'-',color='red',linewidth=5)
pt.plot(x_pred[:-9], anomaly,'o',markersize=20)
pt.show()
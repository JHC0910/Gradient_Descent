import numpy as np
import xlsxwriter
import pandas as pd
import matplotlib.pyplot as plt


x_real = np.random.randint(low = 0.5 ,high = 200.6 ,size = 100)

#y_real = np.exp(-0.6 * x_real) * (-1.789 + 5.875 * x_real - 0.216 * x_real ** 2 + 0.002 * x_real ** 5 ) + 0.52 * x_real ** 3 + 0.005 * np.random.randn(100)
#y_real = 0.385 * ( x_real + np.random.randint(low = 0.5 ,high = 2.0 ,size = 100)) - 1.689 + 0.005 * np.random.randint(low = 0.5 ,high = 2.0 ,size = 100)
y_real = -1.789 + 5.875 * x_real - 0.216 * x_real ** 2 + 2.052 * x_real ** 3 + 0.005 * np.random.randn(100)

cols = ["x" ,"y"]

table = pd.DataFrame([x_real,y_real], cols).T

print(table)

x = np.linspace(0,210,100)
f = lambda x: -1.789 + 5.875 * x - 0.216 * x ** 2 + 2.052 * x ** 3 
y = f(x)

plt.figure(figsize=(12,5))
plt.scatter(x_real ,y_real, label = "y = f(x)")
plt.plot(x,y)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

table.to_csv("generated_data1.csv")


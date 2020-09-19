# Gradient_Descent

## 目的
藉由讀取任意數據並自動進行線性回歸找出數據背後的underlying model

## 測試
* 目標函數:
  * y = 0.385 x - 1.689
利用數據產生器以以下模型產生數據，此模型加入了隨機數使得數據不會太符合目標函數的直線
~~~js
y_real = 0.385 * ( x_real + np.random.randint(low = 0.5 ,high = 2.0 ,size = 100)) - 1.689 + 0.005 * np.random.randint(low = 0.5 ,high = 2.0 ,size = 100)
~~~





## Generalization
雖然是用一維方程式來進行測試，事實上可以




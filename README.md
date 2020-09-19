# Gradient_Descent

## 目的
藉由讀取任意數據並自動進行線性回歸找出數據背後的underlying model

## 測試目標
* 目標函數:
  * y = 0.385 x - 1.689
利用數據產生器以以下模型產生數據，此模型加入了隨機數使得數據不會太符合目標函數的直線
~~~js
y_real = 0.385 * ( x_real + np.random.randint(low = 0.5 ,high = 2.0 ,size = 100)) - 1.689 + 0.005 * np.random.randint(low = 0.5 ,high = 2.0 ,size = 100)
~~~

## 原理
這裡分別使用了單純的python以及利用TensorFlow來進行相同運算

* Python:
~~~js
#target: ydata = w * xdata + b
b = 200
w = 20
lr = 0.00001

b_history = [b]
w_history = [w]

for i in range(100000):
    b_grad = 0.0
    w_grad = 0.0
    
    for n in range(len(x_train)):
        b_grad = b_grad - 2 * (y_train[n] - (w * x_train[n] + b))
        w_grad = w_grad - 2 * (y_train[n] - (w * x_train[n] + b)) * x_train[n]
        
    b = b - lr * b_grad
    w = w - lr * w_grad
    
    b_history.append(b)
    w_history.append(w)

    
[w_history[-1], b_history[-1]]
~~~



## Generalization
雖然是用一維方程式來進行測試，事實上可以直接應用到多變數的情形。
此時只要將X定義為由多變數所組成的向量，之後採用相同原理的運算即可進行回歸。




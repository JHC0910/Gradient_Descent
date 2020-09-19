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
這裡分別使用了單純的python以及利用TensorFlow來進行相同運算。
概念均為先假設一個線性的目標函數，其斜率(在DL中稱kernal)及截距(在DL中稱bias)初始值隨意設。
接著在每次iteration計算一次目標函數對kernal及bias的微分來得到gradient值。
之後此次的kernal及bias減去學習率(learning rate)乘以gradient值成為下一次迭代的kernal及bias。

* **Python**:
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

* **TensorFlow**:
~~~js
class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
          units=1,
          activation=None,
          kernel_initializer=tf.zeros_initializer(),
          bias_initializer=tf.zeros_initializer()
        )
    def call(self, input):
        output = self.dense(input)
        return output
        
model = Linear()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.00001)

for i in range(100000):
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.reduce_mean(tf.square(y_pred - y))  
    grads = tape.gradient(loss, model.variables)

    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
print(model.variables)
~~~

## Result
* **Python**:
  * y = 0.3855 x - 1.5187
* **TensorFlow**:
  * y = 0.3747 x - 1.2586


## Generalization
雖然是用一維方程式來進行測試，事實上可以直接應用到多變數的情形。
此時只要將X定義為由多變數所組成的向量，之後採用相同原理的運算即可進行回歸。




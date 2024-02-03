import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,), activation='linear')
])


model.compile(optimizer='sgd', loss='mean_squared_error')

history = model.fit(X, y, epochs=100, batch_size=10)

new_data = np.array([[2.5]])
predictions = model.predict(new_data)
print(predictions)


plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.show()

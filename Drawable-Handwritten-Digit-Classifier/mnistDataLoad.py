import tensorflow as tf
import numpy as np 
from sklearn.preprocessing import OneHotEncoder

def reshape_2d_to_1d(data, input_shape):
	return tf.reshape(data, [data.shape[0]] + input_shape)

def data_to_float32(data):
	return tf.cast(data, dtype=tf.float32)

def model_struct(input_shape):
	model = tf.keras.Sequential([
		tf.keras.layers.Conv2D(28, (3, 3), input_shape=input_shape),
		tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(128, activation=tf.nn.relu),
		tf.keras.layers.Dense(10, activation=tf.nn.softmax)
		])

	return model

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

input_shape = [28, 28, 1]

X_train = reshape_2d_to_1d(X_train, input_shape)
X_test = reshape_2d_to_1d(X_test, input_shape)

X_train = data_to_float32(X_train)
X_test = data_to_float32(X_test)

X_train /= 255
X_test /= 255

y_train = tf.reshape(y_train, [-1, 1])
y_test = tf.reshape(y_test, [-1, 1])

encoder = OneHotEncoder(sparse=False)

y_train = tf.convert_to_tensor(encoder.fit_transform(y_train))
y_test = tf.convert_to_tensor(encoder.fit_transform(y_test))

model = model_struct(input_shape)

loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

h = model.fit(x=X_train, y=y_train, epochs=20, validation_data=(X_test, y_test), batch_size=64)

model.save('model.h5')
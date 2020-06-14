from tensorflow import keras
model = keras.models.load_model('/Users/a./Desktop/TIME SERIES MODEL')




from tensorflow import keras
import tensorflow as tf
import csv
time_step = []
sunspots = []
import matplotlib.pyplot as plt
import numpy as np

with open('/Users/a./Downloads/Sunspots.csv') as csvfile:
    reader  = csv.reader(csvfile,delimiter=',')
    next(reader)
    for row in reader:
        sunspots.append(float(row[2]))
        time_step.append(float(row[0]))

series  = np.array(sunspots)
time = np.array(time_step)

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

split_time = 3000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 60
batch_size = 32
shuffle_buffer_size = 1000

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset

print(series[0])
print(series[1])
print(series[2])
x = model.predict(series[0:60][np.newaxis])
y = model.predict(series[1:61][np.newaxis])
z = model.predict(series[3191:3251][np.newaxis])

print(x)
print(y)
print(z)

# forecast=[]
# for time in range(100):
#    print(time)
#    forecast.append(model.predict(series[time:time + window_size][np.newaxis]))
#    print(forecast)
# #
# forecast = forecast[split_time-window_size:]
# print(forecast)
# results = np.array(forecast)
# print(results)
# # plt.figure(figsize=(10, 6))
#
# # plot_series(time_valid, x_valid)

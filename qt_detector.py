import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, TimeDistributed, Flatten, LSTM, Bidirectional, Dropout, BatchNormalization, Conv2D, Activation, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import tensorflow as tf
import random as rn

np.random.seed(123)
rn.seed(123)
tf.random.set_seed(123)

#WIN_LEN = 125, CHANNEL(morl 변환을 통해 나온 값)
WIN_LEN = 125
CHANNEL = 62
BATCH_SIZE = 32
EPOCHS = 30   


data_train = pd.read_pickle('train.pkl')
X_train = np.array([np.reshape(x, (WIN_LEN,CHANNEL)) for x in data_train['window'].values])
X_train = np.expand_dims(X_train,3)
y_train = np.array([y for y in data_train['label'].values])
y_train = to_categorical(np.expand_dims(y_train, axis=2))

data_valid = pd.read_pickle('valid.pkl')
X_valid = np.array([np.reshape(x, (WIN_LEN,CHANNEL)) for x in data_valid['window'].values])
X_valid = np.expand_dims(X_valid,3)
y_valid = np.array([y for y in data_valid['label'].values])
y_valid = to_categorical(np.expand_dims(y_valid, axis=2))

data_test = pd.read_pickle('test.pkl')
X_test = np.array([np.reshape(x, (WIN_LEN,CHANNEL)) for x in data_test['window'].values])
X_test = np.expand_dims(X_test,3)
y_test = np.array([y for y in data_test['label'].values])
y_test = to_categorical(np.expand_dims(y_test, axis=2))
data_test = np.array([data for data in data_test['data'].values])


tf.random.set_seed(123)

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=5, padding='same', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(1, 4)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=128, kernel_size=5, padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(1, 4)))
model.add(Dropout(0.25))

model.add(TimeDistributed(Flatten()))
model.add(Bidirectional(LSTM(units=100, return_sequences=True, dropout=0.25)))

model.add(Bidirectional(LSTM(units=50, return_sequences=True, dropout=0.25)))

model.add(TimeDistributed(Dense(10, activation='relu')))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(TimeDistributed(Dense(2, activation='softmax')))

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy'],
)

model.summary()




tf.random.set_seed(123)
if do_training:
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5)

    history = model.fit(
      X_train,
      y_train,
      batch_size=BATCH_SIZE,
      epochs=EPOCHS,
      verbose=1,
      callbacks=[early_stop],
      shuffle=True,
      validation_data=(X_valid, y_valid),
    )
    
    history = history.history
    model.save('model.h5')
    pickle.dump(history, open('history.pkl', 'wb'))

else:
    model = load_model('model.h5')
    history = pickle.load(open('history.pkl', 'rb'))


def roc(predictions, true):
    predictions = predictions.flatten()
    true = true.flatten()

    thresh_vals = [i/25 for i in range(26)]
    results = []
    for thresh in thresh_vals:
        tmp_predictions = (predictions < thresh).astype(int)
        f1 = f1_score(true, tmp_predictions)
        tn, fp, fn, tp = confusion_matrix(true, tmp_predictions).ravel()
        tpr = tp/(tp+fn)
        fpr = fp/(tn+fp)
        acc = (tp+tn)/(tn+fp+fn+tp)

        tmp_dict = {'f1': f1, 'acc': acc, 'tpr': tpr, 'fpr': fpr, 'thresh': thresh}
        results.append(tmp_dict)


    results = pd.DataFrame(results)
    results = results.sort_values(by='thresh', ascending=False)

    results = results.sort_values(by='f1', ascending=False)
    final_thresh = results.head(1)['thresh'].values[0]

    return results, final_thresh


prediction_train = model.predict(X_train)

prediction_train = prediction_train[:,:,0]
y_train = y_train[:,:,1]

results, thresh = roc(prediction_train, y_train)


predictions = model.predict(X_test)

predictions = predictions[:,:,0]
y_test = y_test[:,:,1]

predictions = np.array([(p>thresh).astype(int) for p in predictions])

inds = np.array(range(len(predictions)))
np.random.shuffle(inds)
for i in inds[0:5]:
    y_labels = y_test[i]
    predictions_tmp = 1-predictions[i]
    coefficients = np.reshape(X_test[i], (62, 125))
    data = data_test[i]



predictions = 1-predictions.flatten()
y_test = y_test.flatten()

f1 = f1_score(y_test, predictions)
tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
tpr = tp/(tp+fn)
fpr = fp/(tn+fp)
precision = tp/(tp+fp)
acc = (tp+tn)/(tn+fp+fn+tp)
roc = roc_auc_score(y_test, predictions)

tmp_dict = {'f1': f1, 'acc': acc, 'roc' : roc, 'tpr': tpr, 'fpr': fpr, 'precision':precision}
print(pd.DataFrame([tmp_dict]))






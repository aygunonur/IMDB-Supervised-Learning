import numpy as np
import pandas as pd
import re

BATCH_SIZE = 64
EPOCHS = 5

class WordConverter:
    def __init__(self):
        self.word_set = set()
        
    def __call__(self, review):
        words = re.findall("[a-zA-Z0-9']+", review.lower())
        self.word_set.update(words)
        
        return np.array(words)
        
    def get_word_dict(self):
        return { word: index for index, word in enumerate(self.word_set)}

wc = WordConverter()
df = pd.read_csv('imdb.csv', converters={0: wc})
word_dict = wc.get_word_dict()
rev_word_dict = {index: word for word, index in word_dict.items()}

for i in range(len(df)):
    df.iloc[i, 0] = np.array([word_dict[word] for word in df.iloc[i, 0]])
  

test_zone = int(len(df) * 0.8)
validation_zone = int(test_zone * 0.8)

df_training_dataset = df.iloc[:validation_zone]
df_validation_dataset = df.iloc[validation_zone:test_zone]
df_test_dataset = df.iloc[test_zone:]

df.iloc[df.iloc[:, 1] == 'positive', 1] = 1
df.iloc[df.iloc[:, 1] == 'negative', 1] = 0

def vectorize(iterable, colsize):
    result = np.zeros((len(iterable), colsize), dtype=np.int8)
    for index, values in enumerate(iterable):
        result[index, values] = 1
        
    return result

from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, df, batch_size):
        self.df = df
        self.batch_size = batch_size
        self.indices = np.arange(len(self.df) // self.batch_size, dtype=np.int32)
        
    def __len__(self):
        return len(self.df) // self.batch_size
    
    def __getitem__(self, index):
        start = self.indices[index] * self.batch_size
        stop = (self.indices[index] + 1) * self.batch_size
        
        dataset_x = vectorize(self.df.iloc[start:stop, 0], len(word_dict))
        dataset_y = self.df.iloc[start:stop, 1].to_numpy().astype(np.int8)
        
        return dataset_x, dataset_y
    
    def on_epoch_end(self):
       np.random.shuffle(self.indices)
       
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential(name='IMDB')
model.add(Dense(100, activation='relu', input_dim=len(word_dict), name='Hidden-1'))
model.add(Dense(100, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
hist = model.fit(DataGenerator(df_training_dataset, BATCH_SIZE), validation_data=DataGenerator(df_validation_dataset, BATCH_SIZE), epochs=EPOCHS)

import matplotlib.pyplot as plt

figure = plt.gcf()
figure.set_size_inches((15, 5))
plt.title('Loss - Epoch Graphics')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(range(1, len(hist.history['loss']) + 1), hist.history['loss'])
plt.plot(range(1, len(hist.history['val_loss']) + 1), hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

figure = plt.gcf()
figure.set_size_inches((15, 5))
plt.title('Binary Accuracy - Epoch Graphics')
plt.xlabel('Epoch')
plt.ylabel('Binary Accuracy')
plt.plot(range(1, len(hist.history['binary_accuracy']) + 1), hist.history['binary_accuracy'])
plt.plot(range(1, len(hist.history['val_binary_accuracy']) + 1), hist.history['val_binary_accuracy'])
plt.legend(['Binary Accuracy', 'Validation Binary Accuracy'])
plt.show()

eval_result = model.evaluate(DataGenerator(df_test_dataset, BATCH_SIZE))
for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]} --> {eval_result[i]}')

predict_text = 'the movie was not good. There are many flaws. Players are act badly'
predict_words = re.findall("[a-zA-Z0-9']+", predict_text.lower())
predict_numbers = [word_dict[pw] for pw in predict_words]
predict_data = vectorize([predict_numbers], len(word_dict))
predict_result = model.predict(predict_data)

if predict_result[0][0] > 0.5:
    print('OLUMLU')
else:
    print('OLUMSUZ')
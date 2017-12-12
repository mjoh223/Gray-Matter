import os
import sys
import matplotlib.pyplot as plt
plt.use('Agg')
import numpy
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding

#define sequences to be learned (krp)
BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'train_krp')
embeddings_index = dict()
f = open(os.path.join(GLOVE_DIR, 'psi_krp.fa'))
seqset_krp = []
for line in f:
	a = line
	s = ' '.join([a[i:i+3] for i in range(0, len(a), 3)])
	seqset_krp.append(s)
f.close()
labels_krp = [1]* 998
#define sequences to be learned (smr)
BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'train_smr')
embeddings_index = dict()
f = open(os.path.join(GLOVE_DIR, 'psi_smr.fa'))
seqset_smr = []
for line in f:
	asmr = line
	ssmr = ' '.join([asmr[i:i+3] for i in range(0, len(asmr), 3)])
	seqset_smr.append(ssmr)	
f.close()	
labels_smr = [0]* 232

seqset = seqset_krp + seqset_smr
labels = labels_krp + labels_smr	
	
# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(seqset) #'AAA': #
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_seqs = t.texts_to_sequences(seqset)
#print(encoded_seqs)
# pad documents to a max length of 75 words (This may have to be extended based on the size of sequence inputs)
max_length = 75
padded_seqs = pad_sequences(encoded_seqs, maxlen=max_length, padding='post')
#print(padded_seqs)
# load the whole amino acid embedding into memory http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0141287
BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'biovec')
embeddings_index = dict()
f = open(os.path.join(GLOVE_DIR, 'protvec.txt'))
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training setseq
embedding_matrix = zeros((vocab_size, 100))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

# define model
model = Sequential()
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=75, trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
# fit the model
history = model.fit(padded_seqs, labels, epochs=50, verbose=1)
# evaluate the model
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy.png')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')

loss, accuracy = model.evaluate(padded_seqs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))

#predictions = model.predict(padded_seqs)
#rounded = [round(x[0]) for x in predictions]
#print(rounded)
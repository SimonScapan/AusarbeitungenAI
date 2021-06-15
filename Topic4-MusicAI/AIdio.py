import os                                               
import random
import numpy as np                                      

from sklearn.model_selection import train_test_split

from music21 import *

from keras.models import load_model
from keras.layers import *
from keras.models import *
from keras.callbacks import *
import keras.backend as K



def read_midi(file):
    '''
    *** Function to read MIDI files ***
    
    input: midi file

    process:
        * midi file is parsed
        * checking for different instruments and looping over them
        * check wether an element is a node or a chord

    output: numpy array of nodes and chords
    '''
    
    midi = converter.parse(file)                                # parse the midi file
    instruments = instrument.partitionByInstrument(midi)        # group based on different instruments
    
    for part in instruments.parts:                              # looping over all the instruments

        if 'Piano' in str(part):                                # select elements of only piano

            notes_to_parse = part.recurse()                     # get notes of instrument          
            for element in notes_to_parse:                      # finding whether a particular element is note or a chord
                
                if isinstance(element, note.Note):              # note
                    notes.append(str(element.pitch))

                elif isinstance(element, chord.Chord):          # chord
                    notes.append('.'.join(str(n) for n in element.normalOrder))

    return np.array(notes)




##############################
# import schubert midi files #
##############################
path = 'schubert/' 

def load_midi(path):
    '''
    '''

    files = [i for i in os.listdir(path) if i.endswith(".mid")]        # read all the filenames
    notes_array = np.array([read_midi(path+i) for i in files])         # reading each midi file

    return notes_array


def get_frequent_nodes(notes_array):
    '''
    '''

    notes_ = [element for note_ in notes_array for element in note_]
    freq = dict(Counter(notes_))
    frequent_notes = [note_ for note_, count in freq.items() if count >= 50]
    new_music = []

    for notes in notes_array:
        temp = []
        for note_ in notes:
            if note_ in frequent_notes:
                temp.append(note_)            
            new_music.append(temp)

    new_music = np.array(new_music)

    return new_music

def prepare_sequences(new_music):
    '''
    '''

    no_of_timesteps = 32
    x = []
    y = []

    for note_ in new_music:

        for i in range(0, len(note_) - no_of_timesteps, 1):

            # preparing input and output sequences
            input_ = note_[i:i + no_of_timesteps]
            output = note_[i + no_of_timesteps]
            
            x.append(input_)
            y.append(output)
            
    x=np.array(x)
    y=np.array(y)

    unique_x = list(set(x.ravel()))
    unique_y = list(set(y))

    x_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_x))
    y_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_y)) 

    x_seq = []
    for i in x:
        temp = []
        for j in i:
            # assigning unique integer to every note
            temp.append(x_note_to_int[j])
        x_seq.append(temp)

    x_seq = np.array(x_seq)
    y_seq=np.array([y_note_to_int[i] for i in y])

    return x_seq, y_seq



# train test split
x_tr, x_val, y_tr, y_val = train_test_split(x_seq, y_seq, test_size = 0.2, random_state = 0)

def train_model(x_tr, x_val, y_tr, y_val):
    '''
    '''

    K.clear_session()
    model = Sequential()

    # embedding layer
    model.add(Embedding(len(unique_x), 100, input_length=32,trainable=True)) 
    model.add(Conv1D(64,3, padding='causal',activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPool1D(2))
    model.add(Conv1D(128,3,activation='relu',dilation_rate=2,padding='causal'))
    model.add(Dropout(0.2))
    model.add(MaxPool1D(2))
    model.add(Conv1D(256,3,activation='relu',dilation_rate=4,padding='causal'))
    model.add(Dropout(0.2))
    model.add(MaxPool1D(2))

    #model.add(Conv1D(256,5,activation='relu'))    
    model.add(GlobalMaxPool1D())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(len(unique_y), activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    model.summary()

    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True,verbose=1)
    history = model.fit(np.array(x_tr),np.array(y_tr),batch_size=128,epochs=50, validation_data=(np.array(x_val),np.array(y_val)),verbose=1, callbacks=[mc])
    model = load_model('best_model.h5') # loading best model

    return model


def compose(x_val, no_of_timesteps):
    '''
    '''

    ind = np.random.randint(0,len(x_val)-1)

    random_music = x_val[ind]

    predictions=[]

    for i in range(10):
        random_music = random_music.reshape(1,no_of_timesteps)
        
        prob  = model.predict(random_music)[0] 
        y_pred= np.argmax(prob,axis=0)
        predictions.append(y_pred)
        
        random_music = np.insert(random_music[0],len(random_music[0]),y_pred)
        random_music = random_music[1:]

    x_int_to_note = dict((number, note_) for number, note_ in enumerate(unique_x)) 
    predicted_notes = [x_int_to_note[i] for i in predictions]

    return predicted_notes

def convert_to_midi(prediction_output):
    '''
    '''

    offset = 0
    output_notes = []
    
    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:

        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
            
                cn=int(current_note)
                new_note = note.Note(cn)
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)

            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
            
        # pattern is a note
        else:

            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
    
        # increase offset each iteration so that notes do not stack
        offset += 1
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='music.mid')


convert_to_midi(predicted_notes)
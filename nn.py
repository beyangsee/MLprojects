import numpy as np
import tensorflow as tf
from tensorflow import keras # he has MLP, CNN, RNN, transformers...
from tensorflow.keras import layers

import pickle
import matplotlib.pyplot as plt


# on A4 paper
# change the pair of (fr, to) into an int
def encode_move_to_int(move, n_tiles=14): # takes (source, destination) to an index int
    src, dst = move # source, destination (4,7) 
    if dst > src: # 7 > 4
        dst -= 1 # move (fr, to) back.  
    return src * (n_tiles-1) + dst   # output is an index number = position in an array, 可能的move有src * (n_tiles-1) 种情况

def decode_move_from_int(pos, n_tiles=14): # pos = index number to represent the position, whichever position
    src, dst = np.divmod(pos, n_tiles-1) # np.divmod = devide first element, by second element
    if dst >= src:
        dst += 1
    return src, dst

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def encode_move_to_vec(move, n_tiles=14): # vector list is 1820(14*13) long -- only the move will become 1, the we make the rest become 0
    pos = encode_move_to_int(move, n_tiles=n_tiles)
    vec = np.zeros(n_tiles * (n_tiles-1))
    vec[pos] = 1
    return vec


def decode_move_from_vec(vec, n_tiles=14):
    pos = np.where(vec)[0][0]
    return decode_move_from_int(pos, n_tiles=n_tiles)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def load_data(filename='data.pickle'): # load training data - pickle file
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    # data = np.load(filename, allow_pickle=True)
    return data


def train_test_data(data, split_percent=0.8): # 80% for training, 20% for testing
    n = len(data)
    split = int(n * split_percent)
    data_train = data[:split]
    data_test = data[split:]
    return data_train, data_test


def data_xy_mlp(data):
    X = []
    Y = []
    W = []
    for game in data:
        winner = game[0]
        player = np.array([d[0] for d in game[-1]])
        moves  = np.array([encode_move_to_vec(d[1]) for d in game[-1]])
        states = np.array([d[2] for d in game[-1]])
        x = np.concatenate([player[:,None], states], axis=-1)
        X.append(x)
        Y.append(moves)
        W.append(winner)
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    W = np.array(W)
    return X, Y, W


# for s in range(14):
#     for e in range(14):
#         if s != e:
#             v = encode_move((s,e))
#             s2,e2 = decode_move(v)
#             if s2!= s or e2 != e:
#                 print('error: %d, %d,  %d, %d' % (s,s2,e,e2))
#             else:
#                 print('(%s,%d) works!' % (e,s))


def create_mlp_model(input_shape, output_shape, hidden_layers, hidden_units,
                     loss='categorical_crossentropy',
                     optimizer='adam'):
    # input
    inp = layers.Input(input_shape)

    # mpl layers
    x = layers.Dense(hidden_units, activation='relu')(inp)
    for l in range(hidden_layers):
        x = layers.Dense(units=hidden_units, activation='relu')(x)

    # output
    out = layers.Dense(units=output_shape, activation='softmax')(x)

    # build model
    model = keras.Model(inp, out)
    model.compile(loss=loss, optimizer=optimizer) # optimize 

    return model


# def create_mlp_model_2(hidden_layers, hidden_units, input_shape, loss='sparse_categorical_crossentropy', activation=('relu', 'tanh')):
#     model = keras.Sequential()
#     model.add(layers.Dense(hidden_units, input_shape=input_shape, activation=activation[0]))
#     for l in range(hidden_layers):
#         model.add(layers.Dense(units=hidden_units, activation=activation[1]))
#     model.add(layers.Dense(units=input_shape[-1]-1, activation=activation[1]))
#     model.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate=1e-3))
#     return model


# def create_rnn_model(hidden_units, dense_units, input_shape, activation=('relu', 'tanh')):
#     model = keras.Sequential()
#     model.add(layers.SimpleRNN(hidden_units, input_shape=input_shape, activation=activation[0]))
#     model.add(layers.Dense(units=dense_units, activation=activation[1]))
#     model.compile(loss=keras.losses.mean_squared_error, optimizer='adam')
#     return model

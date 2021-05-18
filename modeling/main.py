from datetime import datetime
import pandas as pd
import os
import util
import modeling.feature_extraction as fe
from modeling import common
from scipy import stats as st
# from sklearn import metrics
import numpy as np
import keras
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.model_selection import ShuffleSplit
import glob

# def f1_score(true, pred):
#     pred = np.where(pred.flatten() > .5, 1, 0)
#     result = metrics.precision_recall_fscore_support(y_true=true,
#                                                      y_pred=pred,
#                                                      average='micro')
#     return result[2]


def correlation(true, pred):
    pred = pred.flatten()
    result = st.pearsonr(true, pred)
    return result[0]


## extra imports to set GPU options
import tensorflow as tf
from keras import backend as k

###################################
# TensorFlow wizardry
config = tf.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5

# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))
###################################

#######		Setting up data		########
train, dev, test = util.train_dev_test_split(util.get_messages())
data = pd.concat([train, test], axis=0)  #excluding dev set from CV
data = data.reset_index(drop=True)
# print(data.head())
########################################

# TODO: this takes a long time --> maybe use shelve?
# https://www.quora.com/What-is-a-fast-efficient-way-to-load-word-embeddings-At-present-a-crude-custom-function-takes-about-3-minutes-to-load-the-largest-GloVe-embedding
embs = common.get_facebook_fasttext_common_crawl(vocab_limit=1_000)

TARGETS = ['empathy', 'distress']

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               min_delta=0,
                                               patience=20,
                                               verbose=0,
                                               mode='auto')

# features_train_centroid=fe.embedding_centroid(train.essay, embs)
# features_train_matrix=fe.embedding_matrix(train.essay, embs, common.TIMESTEPS)

# features_test_centroid=fe.embedding_centroid(test.essay, embs)
# features_test_matrix=fe.embedding_matrix(test.essay, embs, common.TIMESTEPS)

FEATURES_MATRIX = fe.embedding_matrix(data.essay, embs, common.TIMESTEPS)
FEATURES_CENTROID = fe.embedding_centroid(data.essay, embs)


def train_ffn(target):
    print(f'training ffn for {target}')
    rs = ShuffleSplit(n_splits=1, train_size=.8)

    ffn = common.get_ffn(units=[300, 256, 128, 1],
                         dropout_hidden=.5,
                         dropout_embedding=.2,
                         learning_rate=1e-3,
                         problem='regression')

    for i, splits in enumerate(rs.split(data)):
        train, test = splits
        # print(len(train), len(test))

        # for target in TARGETS:
        # print(target)

        labels_train = data[target][train]
        labels_test = data[target][test]

        features_train_centroid = FEATURES_CENTROID[train]
        # features_train_matrix = FEATURES_MATRIX[train]

        features_test_centroid = FEATURES_CENTROID[test]
        # features_test_matrix = FEATURES_MATRIX[test]

        # print(labels_train)
        # print(features_train_matrix)

        ffn.fit(features_train_centroid,
                labels_train,
                epochs=200,
                validation_split=.1,
                batch_size=32,
                callbacks=[early_stopping])

        #	PREDICTION
        pred = ffn.predict(features_test_centroid)

        #	SCORING
        result = correlation(true=labels_test, pred=pred)

        #	RECORD
        print(result)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H.%M')
    print(f'saving ffn for {target}')
    ffn.save(f'./models/ffn_{target}_{timestamp}.h5')

    return ffn


def load_model(target):
    list_of_files = glob.glob(
        f'./models/*{target}*.h5'
    )  # * means all if need specific format then *.csv

    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)

        if latest_file:
            print(f'loading latest saved model: {latest_file}')
            ffn = keras.models.load_model(latest_file)

    else:
        print('training model from scratch')
        ffn = train_ffn(target)

    return ffn


if __name__ == '__main__':

    words = pd.Series([
        "helps", 
        "uncommon", 
        "blank", 
        "iraqis", 
        "explored", 
        "concentrate",
        "fabrication",
        "leukemia",
        "lakota",
        "healing",
        "inhumane",
        "dehumanizes",
        "mistreating"
    ]).rename('words')
    word_centroids = fe.embedding_centroid(words, embs)

    # mark all out of vocab words
    oov = np.where(~word_centroids.any(axis=1))[0]
    words[oov] = '<OOV> ' + words[oov].astype(str) + ' </OOV>'

    for target in TARGETS:
        ffn = load_model(target)
        pred = ffn.predict(word_centroids)

        pred = pd.DataFrame(pred).join(words)

        print(f'predictions for {target}')
        print(pred)
from tensorboard.plugins import hparams
import shutil
from libray import *
import pandas as pd
import numpy as np

import pickle
from sklearn.metrics import classification_report,matthews_corrcoef

import sklearn as skl
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.callbacks import EarlyStopping
np.random.seed(101)
tf.random.set_seed(101)

# print(f"PyTorch Version: {torch.__version__}")
print(f"Pandas  Version: {pd.__version__}")
# print(f"Snorkel Version: {snorkel.__version__}")
print(f"TF2 Version: {tf.__version__}")
print(f"sklearn Version: {skl.__version__}")

def is_gpu_available():
    return tf.test.is_gpu_available()

def store(b, file_name):
    pickle.dump(b, open(file_name, "wb"))


from tensorboard.plugins.hparams import api as hp 
tf.debugging.set_log_device_placement(True)

#folder_name = "arch"
folder_name = "arch_2"

if os.path.isdir(f"../{folder_name}/"):
    shutil.rmtree(f"../{folder_name}/")
else:
    pass 

 
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([32]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.2, 0.5))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd','adamax']))
HP_DENSE_LAYERS = hp.HParam("dense_layers", hp.Discrete([1, 3, 5 ,7, 10]))

# HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([8 ]))
# HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.2, 0.5))
# HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam']))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer(f'../{folder_name}/hparam_tuning').as_default():
    hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER,HP_DENSE_LAYERS],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )


# x_train, y_train,x_val , y_val, x_test , y_test , x_val_bal , y_val_bal , x_test_bal , y_test_bal= read_data()
# x_train, y_train,x_val , y_val, x_test , y_test , x_val_bal , y_val_bal , x_test_bal , y_test_bal, x_ml_classifer,x_ml_classifer_test, x_ml_classifer_bal, x_ml_classifer_test_bal = read_data()
x_train, y_train , x_val , y_val = read_data_gold_data()

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
# early_stop = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=25)

# Check the shape of the data, if data is empty or not, if the data is empty, then the data is not loaded correctly
for dataframe in [x_train, y_train,x_val , y_val]:
    if dataframe.isnull().values.any() == True:
        raise SystemExit(f'Error data empty {dataframe.shape}')
    else:
        pass
    


def build_model_hp(hparams):
    """
    This function is a merge with the hparams from tensorboard and a variable number of layers.
    The models will be stored on an list 
    The num of layers must be a number
    """
    model = Sequential()

    model.add(Dense(  units=40,activation='relu',input_dim=x_train.shape[1]))
    # model_name = ''
    for _ in range(hparams[HP_DENSE_LAYERS]):
        model.add(Dense(units=hparams[HP_NUM_UNITS], activation="relu"))        
        model.add(Dense(units=hparams[HP_NUM_UNITS], activation="relu"))
        model.add(Dropout(hparams[HP_DROPOUT]))
    # for i in range(num_layers):
        # model.add(Dense( units=hparams[HP_NUM_UNITS],activation='relu'))
        # model.add(Dropout(hparams[HP_DROPOUT]))
        # model_name = f"Dense_Dropout_{i}"
    model.add(Dense(units=1,activation='sigmoid'))
    optimizer = hparams[HP_OPTIMIZER]
    model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['acc'])
    return model

def train_test_model(hparams, x_train, y_train, x_val, y_val, logdir, num_units,dropout_rate,optimizer,layers ):
    model = build_model_hp(hparams)
    
    model.fit(x=x_train, 
          y=y_train, 
          epochs=100, # Run with 1 epoch to speed things up for demo purposes
          validation_data=(x_val, y_val), verbose=0,
          callbacks=[early_stop,
          tf.keras.callbacks.TensorBoard(logdir),  # log metrics
          hp.KerasCallback(logdir, hparams)  
                    ]
          )
    print ( model.metrics_names )
    loss , accuracy = model.evaluate(x_val, y_val, verbose=2)
    model.save(f'../models/TF2_models_snorkel_trained_{num_units}_{dropout_rate}_{optimizer}_{layers}.h5')
#     _, mse = model.evaluate(x_test, y_test) 
    return accuracy , loss


# for each run that you could do
 
def run(run_dir, hparams,x_train, y_train, x_val , y_val, num_units,dropout_rate,optimizer ,layers ):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy , loss = train_test_model(hparams, x_train, y_train, x_val , y_val , run_dir, num_units,dropout_rate,optimizer,layers )
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
        tf.summary.scalar('loss', loss, step=1)


### GRID SEARCH 


session_num = 0

for num_units in HP_NUM_UNITS.domain.values:
    for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
        for optimizer in HP_OPTIMIZER.domain.values:
            for layers in HP_DENSE_LAYERS.domain.values :
                hparams = {
            HP_NUM_UNITS: num_units,
            HP_DROPOUT: dropout_rate,
            HP_OPTIMIZER: optimizer,
            HP_DENSE_LAYERS:layers
        }
                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                run(f'../{folder_name}/hparam_tuning/' + run_name, hparams, x_train , y_train , x_val, y_val,num_units,dropout_rate,optimizer,layers)

                session_num += 1

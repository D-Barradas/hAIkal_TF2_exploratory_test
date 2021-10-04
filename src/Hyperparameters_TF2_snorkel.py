import os ,sys 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import pickle

import sklearn as skl
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
import shutil
np.random.seed(101)
tf.random.set_seed(101)


print(f"TF Version: {tf.__version__}")
print(f"Pandas  Version: {pd.__version__}")
print(f"Sklearn Version: {skl.__version__}")
print(f"pickle Version : {pickle.format_version}")


# %%get_ipython().run_line_magic('load_ext', 'tensorboard')
from tensorboard.plugins.hparams import api as hp 


# %%get_ipython().run_line_magic('rm', '-rf ./logs/')
if os.path.isdir("../logs_2/"):
    shutil.rmtree("../logs_2/")
else:
    pass 

# %%
PDB_BM5 = [
'1EXB','1JTD','1M27','1RKE','2A1A','2GAF','2GTP','2VXT','2W9E',
'2X9A','2YVJ','3A4S','3AAA','BAAD','3AAD','3BIW','3BX7',
'3DAW','3EO1','3EOA','3F1P','3FN1','3G6D','3H11',
'3H2V','3HI6','3HMX','3K75','3L5W','3L89','3LVK','3MXW',
'BP57','CP57','3P57','3PC8','3R9A','3RVW','3S9D','3SZK',
'3V6Z','3VLB','4DN4','4FQI','4FZA','4G6J','4G6M','4GAM',
'4GXU','4H03','4HX3','4IZ7','4JCV','4LW4','4M76'
]
selected_features = ['AP_DARS', 'AP_DDG_W', 'AP_DFIRE2', 'AP_GOAP_DF', 'AP_MPS', 'AP_PISA',
       'AP_dDFIRE', 'AlAr', 'ArAr', 'BSA_Apolar', 'CONSRANK_val', 'CP_BFKV',
       'CP_BT', 'CP_D1', 'CP_HLPL', 'CP_MJ2h', 'CP_MJ3h', 'CP_MJPL', 'CP_Qp',
       'CP_RMFCA', 'CP_RMFCEN1', 'CP_RMFCEN2', 'CP_SKOIP', 'CP_SKOb', 'CP_TB',
       'CP_TD', 'CP_TEl', 'CP_TS', 'CP_TSC', 'CP_ZS3DC_MIN', 'DDG_V',
       'PROPNSTS', 'PYDOCK_TOT', 'SIPPER', 'cips_AlAr', 
        'idx', 'RFC', 'TF2','NNC', 'label_binary']
scoring_functions = ['CONSRANK_val', 'CP_HLPL', 'CP_MJ3h', 'DDG_V', 'CP_RMFCA', 'AP_GOAP_DF', 'NNC', 'RFC']

feat_object = ['idx','class_q','label_binary','DQ_val']


def read_data() :
    path1 = "../data/BM5_analysis_balanced_data.csv" 
    path2 = "../data/Clean_dataframe_unbalanced_all_data_ccharppi_4_march_2020_complete_for_snorkel.csv"
    df_set_balanced = pd.read_csv(path1,dtype={'class_q': 'object'})
    df_set_unbalanced = pd.read_csv(path2,dtype={'class_q': 'object'})
    
    ### lets read the scorers set ## 
    # path_3 = "../data/Clean_dataframe_balanced_scorers_set_for_snorkel.csv"
    path_3 = "../data/Scorers_set_analysis_balanced_data.csv"
    path_4 = "../data/Scorers_set_analysis_unbalanced_data.csv"
    # path_4 = "../data/Clean_dataframe_unbalanced_scorers_set_for_snorkel.csv"
    df_scorers_set_balanced = pd.read_csv(path_3)
    df_scorers_set_unbalanced = pd.read_csv(path_4)
    
    df_set_balanced.rename(columns={'NIS Polar' :'NIS_Polar',
                                  'Nis Apolar':'Nis_Apolar',
                                  'BSA Apolar':'BSA_Apolar',
                                  'BSA Polar' :'BSA_Polar'},inplace=True)
    df_set_unbalanced.rename(columns={'NIS Polar' :'NIS_Polar',
                                  'Nis Apolar':'Nis_Apolar',
                                  'BSA Apolar':'BSA_Apolar',
                                  'BSA Polar' :'BSA_Polar'},inplace=True)
    
    df_scorers_set_balanced.rename(columns={'identification' :'idx',
                                  'binary_label':'label_binary'},inplace=True)
    df_scorers_set_unbalanced.rename(columns={'identification' :'idx',
                                  'binary_label':'label_binary'},inplace=True)
    
    df_set_balanced.set_index("Conf",inplace=True)
    df_set_unbalanced.set_index("Conf",inplace=True)
    
    df_scorers_set_balanced.set_index("Conf",inplace=True)
    df_scorers_set_unbalanced.set_index("Conf",inplace=True)
    df_scorers_set_balanced.dropna(inplace=True)
    df_scorers_set_unbalanced.dropna(inplace=True)
    
     ## Split into Training , validation and Testing Sets 
    x_train = df_set_balanced[~df_set_balanced["idx"].isin(PDB_BM5)]
    y_train = df_set_balanced[~df_set_balanced["idx"].isin(PDB_BM5)]["label_binary"].astype("bool")
    x_val = df_set_unbalanced[df_set_unbalanced["idx"].isin(PDB_BM5)]
    y_val = df_set_unbalanced[df_set_unbalanced["idx"].isin(PDB_BM5)]["label_binary"].astype("bool")
    x_test = df_scorers_set_unbalanced
    y_test = df_scorers_set_unbalanced["label_binary"].astype("bool")
    
    ## Select the features for the sets 
    x_train= x_train[scoring_functions] 
#     x_train= x_train.drop(['label_binary','TF2','idx'],axis=1)

    x_val  = x_val[scoring_functions]
#     x_val = x_val.drop(['label_binary','TF2','idx'],axis=1)
    
    x_test = x_test[scoring_functions]
#     x_test = x_test.drop(['label_binary','TF2','idx'],axis=1)
    
    ## Make a copy for handeling better 
    
    x_train = x_train.copy()
    y_train = y_train.copy()
    x_val = x_val.copy()
    y_val = y_val.copy()
    x_test = x_test.copy()
    y_test= y_test.copy()
    
    # scale 
    min_max_scaler = MinMaxScaler()
    for classifier in scoring_functions:
        if classifier != "RFC" or classifier != "NNC":
            x_train[classifier] = min_max_scaler.fit_transform(x_train[classifier].values.reshape(-1,1))
            x_val[classifier]  = min_max_scaler.transform(x_val[classifier].values.reshape(-1,1))
            x_test[classifier]  = min_max_scaler.transform(x_test[classifier].values.reshape(-1,1))
    return x_train, y_train,x_val , y_val, x_test , y_test

x_train, y_train,x_val , y_val, x_test , y_test = read_data()

## redifine x_train and y_train 
## coment this section for normal train

df = pd.read_csv("../data/snorkel_train_gold_set.csv")
df.set_index('Conf',inplace=True)
y_train = df['label_binary'].astype("bool")
x_train = df.drop(['label_binary','TF2','idx'],axis=1)
x_train= x_train[scoring_functions]

# HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([8, 16, 32 ]))
# HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.2, 0.5))
# HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd','adamax']))

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([8 ]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.2, 0.5))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam']))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('../logs_2/hparam_tuning').as_default():
    hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )

def build_model_hp(hparams):
    tf.compat.v1.reset_default_graph()

    model = Sequential()
    model.add(Dense(  units=40,activation='relu',input_dim=x_train.shape[1]))
    model.add(Dense(units=hparams[HP_NUM_UNITS],activation='relu'))
    model.add(Dropout(hparams[HP_DROPOUT]))
    model.add(Dense(units=hparams[HP_NUM_UNITS],activation='relu'))
    model.add(Dense(units=hparams[HP_NUM_UNITS],activation='relu'))
    model.add(Dropout(hparams[HP_DROPOUT]))
    model.add(Dense(units=hparams[HP_NUM_UNITS],activation='relu'))
    model.add(Dense(units=hparams[HP_NUM_UNITS],activation='relu'))
    model.add(Dropout(hparams[HP_DROPOUT]))
    model.add(Dense(units=hparams[HP_NUM_UNITS],activation='relu'))
    model.add(Dense(units=hparams[HP_NUM_UNITS],activation='relu'))
    model.add(Dense(units=1,activation='sigmoid'))
    optimizer = hparams[HP_OPTIMIZER]
    model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['acc'])

    return model


early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)



def train_test_model(hparams, x_train, y_train, x_val, y_val, logdir, num_units,dropout_rate,optimizer ):
    model = build_model_hp(hparams)
    
    model.fit(x=x_train, 
          y=y_train, 
          epochs=50, # Run with 1 epoch to speed things up for demo purposes
          validation_data=(x_val, y_val), verbose=0,
          callbacks=[early_stop,
          tf.keras.callbacks.TensorBoard(logdir),  # log metrics
          hp.KerasCallback(logdir, hparams)  
                    ]
          )

    loss , accuracy = model.evaluate(x_val, y_val, verbose=2)
    model.save(f'../models/TF2_models_snorkel_trained_{num_units}_{dropout_rate}_{optimizer}.h5')
#     _, mse = model.evaluate(x_test, y_test) 
    return accuracy


# for each run that you could do
 
def run(run_dir, hparams,x_train, y_train, x_val , y_val ):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(hparams, x_train, y_train, x_val , y_val , run_dir)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)


### GRID SEARCH 

#print (x_train.shape)
#print (y_train.shape)
session_num = 0

for num_units in HP_NUM_UNITS.domain.values:
    for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
        for optimizer in HP_OPTIMIZER.domain.values:
            hparams = {
          HP_NUM_UNITS: num_units,
          HP_DROPOUT: dropout_rate,
          HP_OPTIMIZER: optimizer,
      }
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            run('../logs_2/hparam_tuning/' + run_name, hparams, x_train , y_train , x_val, y_val,num_units,dropout_rate,optimizer)
            session_num += 1

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os ,sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import pickle
from sklearn.metrics import classification_report,matthews_corrcoef

import sklearn as skl
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
np.random.seed(101)
tf.random.set_seed(101)


# In[2]:


print(f"TF Version: {tf.__version__}")
print(f"Pandas  Version: {pd.__version__}")
print(f"Sklearn Version: {skl.__version__}")
print(f"pickle Version : {pickle.format_version}")


# In[3]:


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



def store(b, file_name):
    pickle.dump(b, open(file_name, "wb"))


# In[26]:


def save_metrics_results(model,x_test,y_test,tag):
    # target_names = ['Incorrect', 'Correct']

#     y_pred = model.predict_classes(x_test.to_numpy())
    y_pred = (model.predict(x_test.to_numpy()) > 0.5).astype("int32")
    cr = classification_report(y_true=y_test, y_pred=y_pred,output_dict=True)
    mmc = matthews_corrcoef(y_true=y_test, y_pred=y_pred)
    # print (cr)
    acc = cr["accuracy"]
    rec_false = cr["False"]["recall"]
    rec_true  = cr["True"]["recall"]
    pres_false = cr["False"]["precision"]
    pres_true = cr["True"]["precision"]
    f1_false =  cr["False"]["f1-score"]
    f1_true =  cr["True"]["f1-score"]

    results = {
        "Accuracy": acc,
        "Recall_inc":rec_false,
        "Recall_cor":rec_true,
        "Precision_inc":pres_false ,
        "Precision_cor":pres_true,
        "F1_inc":f1_false,
        "F1_cor":f1_true,
        "MCC":mmc
    }
    mean_df = pd.DataFrame(data=results,index=[f"{tag}"])
    return mean_df.round(4)


# In[7]:


def build_model():
    tf.compat.v1.reset_default_graph()

    model = Sequential()
    ### this is the original architecture a.k.a V1 
#    model.add(Dense(units=30,activation='relu',input_dim=8))
#    model.add(Dropout(0.5))
#    model.add(Dense(units=15,activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(units=1,activation='sigmoid'))
    ###### 

    model.add(Dense(  units=40,activation='relu',input_dim=x_train.shape[1]))

    model.add(Dense(units=8,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=8,activation='relu'))
    model.add(Dense(units=8,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=8,activation='relu'))
    model.add(Dense(units=8,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=8,activation='relu'))
    model.add(Dense(units=8,activation='relu')


    model.add(Dense(units=1,activation='sigmoid'))
    optimizer = "adam"
    model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['acc'])

    return model


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

x_train, y_train, x_val , y_val, x_test , y_test = read_data()

## redifine x_train and y_train 
## coment this section for normal train

df = pd.read_csv("../data/snorkel_train_gold_set.csv")
df.set_index('Conf',inplace=True)
y_train = df['label_binary'].astype("bool")
x_train = df.drop(['label_binary','TF2','idx'],axis=1
x_train= x_train[scoring_functions]

## train 
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

model = build_model()


model.fit(x=x_train,
          y=y_train,
          epochs=45,
          validation_data=(x_val ,y_val ),
          verbose=1,
          #callbacks=[early_stop]
          )


# In[27]:
model.save('../models/TF2_models_snorkel_trained_v3.h5')

validation_df_results= save_metrics_results(model=model,
                     x_test=x_val,
                     y_test=y_val,
                    tag="validation")



print ( validation_df_results)



test_df_results= save_metrics_results(model=model,
                     x_test=x_test,
                     y_test=y_test,
                    tag="Test")



print ( test_df_results)

res  = pd.concat([validation_df_results,test_df_results])
res.to_csv("../results/results_TF2_snorkel_v2.csv")
# In[ ]:


#model_loss = pd.DataFrame(model.history.history)
#print (model_loss.columns)
#model_loss[["loss","val_loss"]].plot()
#plt.savefig("../results/TF2_training_loss_snorkel_v1.png",transparent=True)
#
#
## In[ ]:
#
#
#model_loss[["acc","val_acc"]].plot()
#plt.savefig("../results/TF2_training_accuracy_snorkel_v1.png",transparent=True)
#
#
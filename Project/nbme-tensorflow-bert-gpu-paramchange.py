import numpy as np 
import pandas as pd
from sklearn.preprocessing import  LabelEncoder
from tqdm.auto import tqdm
import random
import os
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import dill
import tensorflow.keras.backend as K
from tqdm.auto import tqdm
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt
import tensorflow as tf
from transformers import AutoTokenizer, AutoConfig,TFAutoModel
import json

# NEW on TPU in TensorFlow 24: shorter cross-compatible TPU/GPU/multi-GPU/cluster-GPU detection code

try: # detect TPUs
    tpu  = tf.distribute.cluster_resolver.TPUClusterResolver.connect() # TPU detection
    tf.config.experimental_connect_to_cluster(tpu )
    tf.tpu.experimental.initialize_tpu_system(tpu )
    strategy = tf.distribute.TPUStrategy(tpu )
    print('Using TPU')
except ValueError: # detect GPUs
    tpu = None
    strategy = tf.distribute.MirroredStrategy() # for GPU or multi-GPU machines
    #strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
    #strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy() # for clusters of multi-GPU machines

print("Number of accelerators: ", strategy.num_replicas_in_sync)


AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print(f'REPLICAS: {REPLICAS}')

seed=999
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
print('Mixed precision enabled')

# fine-tunes all layers with 10 epochs and low learning rate
TRAIN = True
# fine tunes only last layer for efficiency, TODO
# https://stackoverflow.com/questions/56028464/fine-tuning-last-x-layers-of-bert
LAST_ONLY = False

features = pd.read_csv("./data/features.csv")
patient_notes = pd.read_csv("./data/patient_notes.csv")
test = pd.read_csv("./data/test.csv")
train= pd.read_csv("./data/train.csv")
sample_submission= pd.read_csv("./data/sample_submission.csv")

test = test.merge(patient_notes,on=['case_num','pn_num']).merge(features,on=['case_num','feature_num'])
train = train.merge(patient_notes,on=['case_num','pn_num']).merge(features,on=['case_num','feature_num'])

train.head(5)

MODEL_NAME = 'bert-base-cased'
DATA_PATH = "./bert_input"
DATA_EXISTS = os.path.exists(DATA_PATH)
#DATA_EXISTS = False
SEQUENCE_LENGTH = 512

if DATA_EXISTS:
    tokenizer = AutoTokenizer.from_pretrained(DATA_PATH+"/my_tokenizer/",normalization=True)
    config = AutoConfig.from_pretrained(DATA_PATH+"/my_tokenizer/config.json")
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,normalization=True)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained('my_tokenizer')
    config.save_pretrained('my_tokenizer')

    EMPTY =  'EMPTY'
CLASSES = [EMPTY,]+features.feature_num.unique().tolist()

if DATA_EXISTS:
    label_encoder = dill.load(open(DATA_PATH+"/label_encoder.dill",'rb'))
else:
    # label_encoder
    label_encoder = LabelEncoder()
    # Encode labels
    label_encoder.fit(CLASSES)
    dill.dump(label_encoder,open('label_encoder.dill','wb'))
# TODO assign some test_data from train
train['TARGET']= label_encoder.transform(train['feature_num'])
test['TARGET']= label_encoder.transform(test['feature_num'])
print(test)
N_CLASSES = len(label_encoder.classes_)
EMPTY_IDX = label_encoder.transform([EMPTY,]) [0]

def decode_location(locations):
    for x in ["[","]","'"]:
        locations = locations.replace(x,'')
    locations = locations.replace(',',';')
    locations = locations.split(";")
    res = []
    for location in locations:
        if location:
            x,y = location.split()
            res.append((int(x),int(y)))
    return sorted(res,key=lambda x:x[0])
    

    if DATA_EXISTS:
        sequences = np.load(open(DATA_PATH+"/sequences.npy",'rb'))
        masks = np.load(open(DATA_PATH+"/masks.npy",'rb'))
        labels = np.load(open(DATA_PATH+"/labels.npy",'rb'))
    else:
        sequences, labels, masks = [], [], []
        for g1 in tqdm(train.groupby('pn_num')):
            gdf = g1[1]
            pn_history  = gdf.iloc[0].pn_history

            tokens = tokenizer.encode_plus(pn_history, max_length=SEQUENCE_LENGTH, padding='max_length',truncation=True, return_offsets_mapping=True)
            sequence = tokens['input_ids']
            attention_mask = tokens['attention_mask']
            label = np.array([EMPTY_IDX for _ in range(SEQUENCE_LENGTH)])

            # BUILD THE TARGET ARRAY
            offsets = tokens['offset_mapping']
            label_empty = True
            for index, row in gdf.iterrows():
                TARGET = row.TARGET
                for i, (w_start, w_end) in enumerate(offsets):
                    for start,end in decode_location(row.location):
                        if w_start < w_end and (w_start >= start) and (end >= w_end):
                            label[i] = TARGET
                            label_empty = False
                        if w_start >= w_end:
                            break
            if not label_empty:
                print(sequence)
                sequences.append(sequence)
                masks.append(attention_mask)
                labels.append(label)

        sequences = np.array(sequences).astype(np.int32)
        masks = np.array(masks).astype(np.uint8)
        labels = np.array(tf.keras.utils.to_categorical(labels,N_CLASSES)).astype(np.uint8)

        np.save(open("sequences.npy",'wb'), sequences)
        np.save(open("masks.npy",'wb'), masks)
        np.save(open("labels.npy",'wb'), labels)

def build_model():
    
    tokens = tf.keras.layers.Input(shape=(SEQUENCE_LENGTH,), name = 'tokens', dtype=tf.int32)
    attention = tf.keras.layers.Input(shape=(SEQUENCE_LENGTH,), name = 'attention', dtype=tf.int32)
    
    if DATA_EXISTS:
        config = AutoConfig.from_pretrained(DATA_PATH+"/my_tokenizer/config.json")
        backbone = TFAutoModel.from_config(config)
    else:
        config = AutoConfig.from_pretrained(MODEL_NAME)
        backbone = TFAutoModel.from_pretrained(MODEL_NAME,config=config)
    
    out = backbone(tokens, attention_mask=attention)[0]
    out = tf.keras.layers.Dropout(0.2)(out)
    out = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(out)
    
    model = tf.keras.Model([tokens,attention],out)
    
    return model

#From TensorFlow 2.11 onwards, the only way to get GPU support on Windows is to use WSL2.
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# Include the epoch in the file name (uses `str.format`)

# Saving weights: https://medium.com/analytics-vidhya/tensorflow-2-0-save-and-restore-models-4708ed3f0d8
checkpoint_path = "bert_training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


if TRAIN:
    # creating a model under the TPUStrategy will place the model in a replicated (same weights on each of the cores)
    # manner on the TPU and will keep the replica weights in sync by adding appropriate collective communications 
    # (all reducing the gradients).
    with strategy.scope():
        model = build_model()

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss',mode='min', patience=3)
        # Create a callback that saves the model's weights every 5 epochs
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, 
            verbose=1, 
            save_weights_only=True,
            period=5)# previously, there was no period defined 

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                      loss=tf.keras.losses.categorical_crossentropy,metrics=['acc',])

        ''' More extensive .fit function call:
            history = model.fit(
            x={'input_ids': x['input_ids'], 'attention_mask': x['attention_mask']},
            #x={'input_ids': x['input_ids']},
            y={'outputs': train_y},
            validation_split=0.1,
            batch_size=32,
            epochs=1)
        '''
        history = model.fit((sequences,masks),labels,
                            batch_size=12,
                            epochs=10,
                            callbacks=[callback,cp_callback])
        # validation_data=(x_val, y_val),

        model.save_weights(f'model.h5')
    # Display the model's architecture
    # model.summary()

if not TRAIN:
    model = build_model()
    #model.load_weights(DATA_PATH+"/model.h5")

test_sequences, test_masks, test_offsets = [], [],[]
row_ids = []
targets = []

for g1 in tqdm(test.groupby('pn_num')):
    gdf = g1[1]
    pn_history  = gdf.iloc[0].pn_history
    targets.append([])
    row_ids.append([])
    
    test_tokens = tokenizer.encode_plus(pn_history, max_length=SEQUENCE_LENGTH, padding='max_length',truncation=True, return_offsets_mapping=True)
    test_sequence = test_tokens['input_ids']
    test_attention_mask = test_tokens['attention_mask'] 

    # BUILD THE TARGET ARRAY
    offset = test_tokens['offset_mapping']
    
    for index, row in gdf.iterrows():
        targets[-1].append(row.TARGET)
        row_ids[-1].append(row.id)
         
    test_sequences.append(test_sequence)
    test_masks.append(test_attention_mask)
    test_offsets.append(offset)

test_sequences = np.array(test_sequences).astype(np.int32)
test_masks = np.array(test_masks).astype(np.uint8)
targets_to_row_ids = [dict(zip(a,b)) for a,b in zip(targets,row_ids)]

preds = model.predict((test_sequences,test_masks),batch_size=16)
preds = np.argmax(preds,axis=-1)


def decode_position(pos):
    return ";".join([" ".join(np.array(p).astype(str)) for p in pos])


def translate(preds,targets_to_row_ids,offsets):
    all_ids = []
    all_pos = []

    for k in range(len(preds)):
        offset = offsets[k]
        pred = preds[k]
        targets_to_ids = targets_to_row_ids[k]
        
        prediction = {targets_to_ids[t]:[] for t in targets_to_ids}
        i = 0
        while i<SEQUENCE_LENGTH:
            label = pred[i]
            
            if label == EMPTY_IDX:
                i += 1
                continue
            if label in targets_to_ids:
                key = targets_to_ids[label]
                start = offset[i][0]
                while i<SEQUENCE_LENGTH:
                    if pred[i] != label:
                        break
                    else:
                        end = max(offset[i])
                    i += 1
                if  end == 0:
                    break
                prediction[key].append((start,end))
            else:
                i+=1
        for key in prediction:
            all_ids.append(key)
            all_pos.append(decode_position(prediction[key]))
    df = pd.DataFrame({
        "id":all_ids,
        "location": all_pos
    })
    return df

sub = translate(preds,targets_to_row_ids,test_offsets)
sub.to_csv('submission.csv',index=False)
sub.head(50)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                      loss=tf.keras.losses.categorical_crossentropy,metrics=['acc',])
# TODO: check if this is the right test data
model.evaluate(x = preds, y = labels, batch_size=12)

'''
evaluate(
    x=None,
    y=None,
    batch_size=None,
    verbose='auto',
    sample_weight=None,
    steps=None,
    callbacks=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
    return_dict=False,
    **kwargs
)
        More extensive .fit function call:
            history = model.fit(
            x={'input_ids': x['input_ids'], 'attention_mask': x['attention_mask']},
            #x={'input_ids': x['input_ids']},
            y={'outputs': train_y},
            validation_split=0.1,
            batch_size=32,
            epochs=1)

        history = model.fit((sequences,masks),labels,
                            batch_size=12,
                            epochs=10,
                            callbacks=[callback,])
'''

print(np.array(test_sequences).shape)
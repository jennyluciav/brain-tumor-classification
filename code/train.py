# -*- coding: utf-8 -*-
from se_resnet import  *
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import SGD, Adam
from keras.models import Model, load_model
from se_inception_resnet_v2 import *
from sklearn import metrics

model  = SEResNet101(classes = 3)

model.load_weights('modified-classifier-101.h5')




gc.collect()
model.summary()

adam = Adam(lr=0.005)
sgd = SGD(momentum=0.9, nesterov=True)
model.compile(optimizer = adam, 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy','categorical_crossentropy', 'categorical_accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-101-lr01-sgd-adam-tumor-classifier-aug-rlr.h5', verbose=1, save_best_only=True)
tensorBoard = TensorBoard(log_dir='./Graph/withaugmentation_resnet101_lr01_sgd_adam_epoch30-rlr', histogram_freq=0,
                            write_graph=True, write_images=True)

results = model.fit(data[:7000], labels[:7000], 
                    validation_split=0.10, 
                    batch_size = 5, epochs=30,
                    callbacks=[earlystopper, checkpointer, tensorBoard, reduce_lr])

model.save('modified-classifier-101-lr01-sgd-adam-aug-rlr.h5')

X_g = X_g[:,:,:,np.newaxis]

X_m = X_m[:,:,:,np.newaxis]

X_p = X_p[:,:,:,np.newaxis]

model.load_weights('modified-classifier-101-lr01-sgd-aug.h5')

model.evaluate(x=X_g, y=y_g, batch_size=7, verbose=1, sample_weight=None, steps=None)
#[0.19074830275386845, 0.9361851364583154, 0.18285635938646444] ---- model-101-lr005-tumor-classifier.h5
#[0.09307191406048802, 0.9789621330912618, 0.08515097974519108] ---- model-tumor-classifier-101.h5
#[0.046026485401948945, 0.9866760176244658, 0.03766815782648718] --- modified-classifier-101.h5
#[0.16412936110535395, 0.9396914481949505, 0.15518619035052506] ----- modified-classifier-101-lr005-bs7

model.evaluate(x=X_m, y=y_m, batch_size=7, verbose=1, sample_weight=None, steps=None)
#[0.5288276857417603, 0.7853107442290096, 0.5209357440129638] ------- model-101-lr005-tumor-classifier.h5
#[0.6451452089221044, 0.7189265641536416, 0.6372242540299977] ---- model-tumor-classifier-101.h5
#[0.25353941822813897, 0.9180791006334084, 0.24518109206977598] ----- modified-classifier-101.h5
#[0.702659157507311, 0.7217514242156077, 0.6937159732178821]  ----- modified-classifier-101-lr005-bs7

model.evaluate(x=X_p, y=y_p, batch_size=7, verbose=1, sample_weight=None, steps=None)
#[0.28449976267343985, 0.9081967263273855, 0.2766078167355777] ----- model-101-lr005-tumor-classifier.h5
#[0.33034841984998986, 0.9114754148845464, 0.32242748550867123] ----model-tumor-classifier-101.h5
#[0.2671060624944626, 0.9103825190028206, 0.258747734994083] ----- modified-classifier-101.h5
#[0.2324212703711348, 0.9311475446315411, 0.2234780984375972] ----- modified-classifier-101-lr005-bs7

model.metrics_names
# ['loss', 'acc', 'categorical_crossentropy']


# overall accuracy of 101 lr005 model = 87.6533.

model.evaluate(x=data[12000:], y=labels[12000:], batch_size=5, verbose=1, sample_weight=None, steps=None)
# [0.31230537256447566, 0.8846503241591622, 0.30157377932685514] ----- modified-classifier-101-lr005-aug.h5
# [0.3039362240027802, 0.8973660366601234, 0.2947890567493885] --- accuracy of augmented model ------- model-101-lr005-tumor-classifier-aug.h5

model.evaluate(x=data, y=labels, batch_size=5, verbose=1, sample_weight=None, steps=None)
# [0.24094252673716135, 0.9190138204014854, 0.23179535953508293] ---- model-101-lr005-tumor-classifier-aug.h5

model.evaluate(x=X_g, y=y_g, batch_size=10, verbose=1, sample_weight=None, steps=None)
#[0.20648954605858716, 0.9326788179503249, 0.19734237907503518] ---- model-101-lr005-tumor-classifier-aug.h5

model.evaluate(x=X_m, y=y_m, batch_size=10, verbose=1, sample_weight=None, steps=None)
#[0.26229804178626187, 0.8954802206007101, 0.25315087436998296] ---- model-101-lr005-tumor-classifier-aug.h5

model.evaluate(x=X_p, y=y_p, batch_size=10, verbose=1, sample_weight=None, steps=None)
#[0.2928584333651704, 0.9103825085801505, 0.28212684412268013] ---- modified-classifier-101-lr005-aug.h5


#augemnted data part 2 evaluation on 'modified-classifier-101-lr01-sgd-adam-aug-rlr.h5'
model.evaluate(x=data[7000:], y=labels[7000:], batch_size=10, verbose=1, sample_weight=None, steps=None) 
#[0.4772191370079157, 0.8460471485424967, 0.4688608170191185, 0.8460471485424967]

model.evaluate(x=X_g, y=y_g, batch_size=10, verbose=1, sample_weight=None, steps=None) 
# [1.957969096440921, 0.42847125294917426, 1.9454682119671376, 0.42847125294917426]

model.evaluate(x=X_m, y=y_m, batch_size=10, verbose=1, sample_weight=None, steps=None) 
# [0.8148369090578987, 0.65254237624885, 0.802336080389171, 0.65254237624885]

model.evaluate(x=X_p, y=y_p, batch_size=10, verbose=1, sample_weight=None, steps=None) 
# [0.08919705322287122, 0.9814207641804804, 0.07669621075076036, 0.9814207641804804]



# model-101-lr01-sgd-adam-tumor-classifier-aug-rlr.h5
model.evaluate(x=data[7000:], y=labels[7000:], batch_size=10, verbose=1, sample_weight=None, steps=None) 
#[0.29126289498252117, 0.8876560255161767, 0.28221489516462767, 0.8876560255161767]

model.evaluate(x=X_g, y=y_g, batch_size=10, verbose=1, sample_weight=None, steps=None) 
#[0.1425134599114785, 0.9474053260403964, 0.1334654623708282, 0.9474053260403964]

model.evaluate(x=X_m, y=y_m, batch_size=10, verbose=1, sample_weight=None, steps=None) 
#[0.29812707974033503, 0.8813559261419005, 0.2890790784086948, 0.8813559261419005]

model.evaluate(x=X_p, y=y_p, batch_size=10, verbose=1, sample_weight=None, steps=None) 
#[0.4912716586733125, 0.8142076448990347, 0.4822236520589375, 0.8142076448990347]



# modified-classifier-101.h5
model.evaluate(x=data[7000:], y=labels[7000:], batch_size=10, verbose=1, sample_weight=None, steps=None) 
# [1.1702851471910964, 0.6574202553987833, 1.1577843055033585, 0.6574202553987833]

model.evaluate(x=X_g, y=y_g, batch_size=10, verbose=1, sample_weight=None, steps=None) 
# [1.957969096440921, 0.42847125294917426, 1.9454682119671376, 0.42847125294917426]

model.evaluate(x=X_m, y=y_m, batch_size=20, verbose=1, sample_weight=None, steps=None) 
# [0.8148369090578987, 0.65254237624885, 0.802336080389171, 0.65254237624885]

model.evaluate(x=X_p, y=y_p, batch_size=10, verbose=1, sample_weight=None, steps=None) 
# [0.08919705322287122, 0.9814207641804804, 0.07669621075076036, 0.9814207641804804]



# -----------------Prediction on origibnal dataset on modified-classifier-101.h5------------------
y_pred = model.predict(X, batch_size=10, verbose=0, steps=None)

matrix = metrics.confusion_matrix(Y.argmax(axis=1), y_pred.argmax(axis=1))

#1407	9	10
#16	650	42
#52	30	833

#Overall = 0.9478517546736634
#Glioma = 0.9866760168302945
#Meningioma = 0.9180790960451978
#Pituitary = 0.9103825136612022



#------------------------------------------------------------------------------------------------------------

model  = SEInceptionResNetV2(classes = 3)

gc.collect()
model.summary()

adam = Adam(lr=0.005)
sgd = SGD(momentum=0.9, nesterov=True)
model.compile(optimizer = sgd, 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy','categorical_crossentropy', 'categorical_accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=3, min_lr=0.001)
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-resnet-inception-lr01-sgd-tumor-classifier-aug-rlr.h5', verbose=1, save_best_only=True)
tensorBoard = TensorBoard(log_dir='./Graph/withaugmentation_resnet-inception_lr01_sgd_epoch30-rlr', histogram_freq=0,
                            write_graph=True, write_images=True )
results = model.fit(data[:7000], labels[:7000], 
                    validation_split=0.10, 
                    batch_size = 5, epochs=30,
                    callbacks=[earlystopper, checkpointer, tensorBoard, reduce_lr])


#---------------------------------------------Extracting Images-------------------------------------------------------





































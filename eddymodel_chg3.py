# importing
import tensorflow
import keras
import numpy as np
import h5py
import hdf5storage
import math
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import mean_squared_error
from keras.layers import Input, Flatten
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
from matplotlib.backends.backend_pdf import PdfPages

# data preprocessing
# filepath = r'C:\Users\Tony\Documents\Deep Learning\eddyconvlstm\venv\twoyeardowndata.mat'
filepath = r'C:\Users\Tony\Documents\Deep Learning\eddyconvlstm\venv\ufiveyeardata.mat'
filepath2 = r'C:\Users\Tony\Documents\Deep Learning\eddyconvlstm\venv\vfiveyeardata.mat'
#mat = hdf5storage.loadmat('twoyeardowndata.mat')
mat = hdf5storage.loadmat('ufiveyeardata.mat')
newdata1 = mat['newnewdata']
newdata1 = newdata1.astype(np.float64)
newdata1 = newdata1[2:260]
mat2 = hdf5storage.loadmat('vfiveyeardata.mat')
newdata2 = mat2['newnewdata']
newdata2 = newdata2.astype(np.float64)

testScores = []
corrScores = []
testScoresv = []
corrScoresv = []
testScores2 = []
corrScores2 = []


# combine
newdata = np.stack((newdata1[:,:,:,0],newdata2[:,:,:,0]),axis=3)


# newdata = np.reshape(data, (data.shape[3],data.shape[0],data.shape[1],data.shape[2]))

mynorm = plt.Normalize(vmin=-1, vmax=1)
tester = np.array(newdata[0,:,:,0])
plt.imshow(tester, vmin=-1,vmax=1);
plt.colorbar()
plt.show()


#newdata = np.reshape(newdata, (newdata.shape[0],1,newdata.shape[1],newdata.shape[2],newdata.shape[3]))
'''
tester = np.array(newdata[0,0,:,:,0])
plt.imshow(tester);
plt.colorbar()
plt.show()
'''

#newdata_min = newdata.min(axis=(0,1,2), keepdims=True)
#newdata_max = newdata.max(axis=(0,1,2), keepdims=True)
newdata_mean = newdata.mean(axis=(0,1,2,3), keepdims=True)
newdata_std = newdata.std(axis=(0,1,2,3), keepdims=True)


#newdata = (newdata - newdata_min)/(newdata_max-newdata_min)
newdata = (newdata - newdata_mean)/newdata_std
ttmean =newdata.mean(axis=(0,1,2), keepdims=True)
ttmin =newdata.min(axis=(0,1,2), keepdims=True)
ttmax =newdata.max(axis=(0,1,2), keepdims=True)
ttsd =newdata.std(axis=(0,1,2), keepdims=True)



trainsize = int(newdata.shape[0]*0.9);
testsize = newdata.shape[0] - trainsize;
train = newdata[0:trainsize,:,:,:]
test = newdata[trainsize:newdata.shape[0],:,:,:]

xtrain = train[0:train.shape[0]-1,:,:,:]
xtrain = np.reshape(xtrain,(xtrain.shape[0],1,xtrain.shape[1],xtrain.shape[2],xtrain.shape[3]))

ytrain = train[1:train.shape[0],:,:,:]
ytrain = np.reshape(ytrain,(ytrain.shape[0],1,ytrain.shape[1],ytrain.shape[2],ytrain.shape[3]))

xtest = test[0:test.shape[0]-1,:,:,:]
xtest = np.reshape(xtest,(xtest.shape[0],1,xtest.shape[1],xtest.shape[2],xtest.shape[3]))

ytest = test[1:test.shape[0],:,:,:]
ytest = np.reshape(ytest,(ytest.shape[0],1,ytest.shape[1],ytest.shape[2],ytest.shape[3]))

tester = np.array(xtest[0,0,:,:,0])
plt.imshow(tester,vmin=-1,vmax=1);
plt.colorbar()
plt.show()



# build model
# data is a {length, width, q, time}
model = Sequential()

model.add(ConvLSTM2D(filters=32, kernel_size=(5, 5),
                   input_shape=(None, 201, 201, 2),
                   padding='same', return_sequences=True, activation='tanh',
                    go_backwards=True,
                     kernel_initializer='glorot_uniform',recurrent_initializer='orthogonal',
                     recurrent_activation='hard_sigmoid', unit_forget_bias=True,
                     dropout=0.4, recurrent_dropout=0.2))

model.add(BatchNormalization())

model.add(ConvLSTM2D(filters = 2, kernel_size=(3, 3),
                   padding='same', return_sequences=True, activation='tanh',
                     go_backwards=True,
                     kernel_initializer='glorot_uniform',recurrent_initializer='orthogonal',
                     recurrent_activation='hard_sigmoid', unit_forget_bias=True,
                     dropout=0.3, recurrent_dropout=0.2))

model.add(BatchNormalization())
'''
model.add(ConvLSTM2D(filters = 5, kernel_size=(3, 3),
                   padding='same', return_sequences=True, activation='tanh',
                     go_backwards=True,
                     kernel_initializer='glorot_uniform',recurrent_initializer='orthogonal',
                     recurrent_activation='hard_sigmoid', unit_forget_bias=True,
                     dropout=0.3, recurrent_dropout=0.2))

model.add(BatchNormalization())
'''
#model.add(Dense(1))

adam = optimizers.adam(lr=0.0001)
model.compile(loss='mean_squared_error', optimizer=adam)



# train model
model.fit(xtrain, ytrain, batch_size= 10,
        epochs = 10, validation_split=0.05)

# make predictions
# trainPredict = model.predict(xtrain,batch_size=2)
testPredicti = model.predict(xtest,batch_size=32,verbose = 1)
# invert predictions
#diff = newdata_max - newdata_min
diff = newdata_std
newdata_min = newdata_mean
# trainPredict = trainPredict*diff + newdata_min
ytrain = ytrain*diff + newdata_min

testPredict = testPredicti*diff + newdata_min
ytest = ytest*diff + newdata_min
# calculate root mean squared error
#for x in range(10):
#    trainScore = math.sqrt(mean_squared_error(ytrain[x,0,:,:,0], trainPredict[x,0,:,:,0]))
#    print('Train Score: %.4f RMSE' % (trainScore))
for x in range(20):
    testScore = math.sqrt(mean_squared_error(ytest[x,0,:,:,0], testPredict[x+1,0,:,:,0]))
    testScores.append(testScore)
    corrScore = np.corrcoef((ytest[x,0,:,:,0].flatten(), testPredict[x+1,0,:,:,0].flatten()))
    corrScores.append(corrScore[0,1])
    print('Test Score: %.4f RMSE' % (testScore))

''''
trainCorr = np.corrcoef((ytrain[0,0,:,:,0], trainPredict[0,0,:,:,0]))
print('Train Correlation Coeff: %.4f corrcoeff' % (trainCorr))
testCorr = np.corrcoef((ytest[0,0,:,:,0], testPredict[0,0,:,:,0]))
print('Test Correlation Coeff: %.4f corrcoeff' % (testCorr))
'''
testScoresPersist = []
corrScoresPersist = []
for x in range(20):
    testScorePersist = math.sqrt(mean_squared_error(ytest[0,0,:,:,0], testPredict[x+1,0,:,:,0]))
    testScoresPersist.append(testScorePersist)
    corrScorePersist = np.corrcoef((ytest[0,0,:,:,0].flatten(), testPredict[x+1,0,:,:,0].flatten()))
    corrScoresPersist.append(corrScorePersist[0,1])

for x in range(20):
    testScorev = math.sqrt(mean_squared_error(ytest[x,0,:,:,1], testPredict[x+1,0,:,:,1]))
    testScoresv.append(testScorev)
    corrScorev = np.corrcoef((ytest[x,0,:,:,1].flatten(), testPredict[x+1,0,:,:,1].flatten()))
    corrScoresv.append(corrScorev[0,1])
    print('Test Score: %.4f RMSE' % (testScorev))
''''
trainCorr = np.corrcoef((ytrain[0,0,:,:,0], trainPredict[0,0,:,:,0]))
print('Train Correlation Coeff: %.4f corrcoeff' % (trainCorr))
testCorr = np.corrcoef((ytest[0,0,:,:,0], testPredict[0,0,:,:,0]))
print('Test Correlation Coeff: %.4f corrcoeff' % (testCorr))
'''
testScoresPersist2 = []
corrScoresPersist2 = []
for x in range(20):
    testScorePersist2 = math.sqrt(mean_squared_error(ytest[0,0,:,:,1], testPredict[x+1,0,:,:,1]))
    testScoresPersist2.append(testScorePersist2)
    corrScorePersist2 = np.corrcoef((ytest[0,0,:,:,1].flatten(), testPredict[x+1,0,:,:,1].flatten()))
    corrScoresPersist2.append(corrScorePersist2[0,1])


# plotting
# plot RMSE
pp = PdfPages("multipage2.pdf")
weeks = np.arange(1,21,1)
fig, ax = plt.subplots()
fig.set_size_inches(10, 5)
ax.plot(weeks, np.array(corrScoresPersist),'g',label='Correlation')
ax.plot(weeks, np.array(testScoresPersist),'r',label='RMSE')
ax.legend(loc='upper left')
ax.set_ylim(0, 1)
ax.set_xlim(1, 10)
ax.set(xlabel='week',
       title='Correlation and RMSE Persistence')
ax.grid()
fig.savefig(pp, format='pdf')
plt.show()


weeks = np.arange(1,21,1)
fig, ax = plt.subplots()
fig.set_size_inches(10, 5)
ax.plot(weeks, np.array(corrScores),'g',label='Correlation')
ax.plot(weeks, np.array(testScores),'r',label='RMSE')
ax.legend(loc='upper left')
ax.set_ylim(0, 1)
ax.set_xlim(1, 10)
ax.set(xlabel='week',
       title='Correlation and RMSE u')
ax.grid()
fig.savefig(pp, format='pdf')
plt.show()

weeks = np.arange(1,21,1)
fig, ax = plt.subplots()
fig.set_size_inches(10, 5)
ax.plot(weeks, np.array(corrScoresv),'g',label='Correlation')
ax.plot(weeks, np.array(testScoresv),'r',label='RMSE')
ax.legend(loc='upper left')
ax.set_ylim(0, 1)
ax.set_xlim(1, 10)
ax.set(xlabel='week',
       title='Correlation and RMSE v')
ax.grid()
fig.savefig(pp, format='pdf')
plt.show()


fig, ax = plt.subplots()
q = ax.quiver(testPredict[10,0,:,:,0],testPredict[10,0,:,:,1],color = 'blue')
fig.savefig(pp, format='pdf')
plt.show()

fig, ax = plt.subplots()
q = ax.quiver(ytest[9,0,:,:,0],ytest[9,0,:,:,1],color = 'blue')
fig.savefig(pp, format='pdf')
plt.show()



''''
weeks = np.arange(1,11,1)
fig, ax = plt.subplots()
ax.plot(weeks, np.array(testScores))
ax.set(xlabel='week', ylabel='error',
       title='RMSE')
ax.grid()
#fig.savefig(pp, format='pdf')
plt.show()

weeks = np.arange(1,11,1)
fig, ax = plt.subplots()
ax.plot(weeks, np.array(corrScores))
ax.set(xlabel='week', ylabel='corr',
       title='Correlations')
ax.grid()
#fig.savefig(pp, format='pdf')
plt.show()
'''
print(newdata_min)
tester = np.array(ytest[5,0,:,:,0])
plt.imshow(tester,vmin=-1,vmax=1);
plt.colorbar()
plt.show()

# newytest = np.reshape(ytest, (ytest.shape[1],ytest.shape[2],ytest.shape[3],ytest.shape[0]))
tester2 = np.array(testPredict[5,0,:,:,0])
plt.imshow(tester2,vmin=-1,vmax=1);
plt.colorbar()
plt.show()

print(newdata_min)
tester = np.array(ytest[1,0,:,:,0])
plt.imshow(tester,vmin=-1,vmax=1);
plt.colorbar()
plt.show()

# newytest = np.reshape(ytest, (ytest.shape[1],ytest.shape[2],ytest.shape[3],ytest.shape[0]))
tester2 = np.array(testPredict[1,0,:,:,0])
plt.imshow(tester2,vmin=-1,vmax=1);
plt.colorbar()
plt.show()

##Plot into PDF file
def visualize(r,obser, predit,var):
    """Draws original, encoded and decoded images"""


    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(10, 10)
    fig.subplots_adjust(wspace=0.5)
    fig.suptitle("Week " +  str(r+1) + " Plots", fontsize=16)

    im1 = ax1.imshow(obser, vmin=-1, vmax=1)
    ax1_divider = make_axes_locatable(ax1)
    cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
    cb1 = colorbar(im1, cax=cax1)
    ax1.set_title("Observed" + var)

    im2 = ax2.imshow(predit, vmin=-1, vmax=1)
    ax2_divider = make_axes_locatable(ax2)
    cax2 = ax2_divider.append_axes("right", size="7%", pad="2%")
    cb2 = colorbar(im2, cax=cax2)
    ax2.set_title("Predicted" + var)

    fig.savefig(pp, format='pdf')





#pp = PdfPages("multipage.pdf")
for i in range(20):
    var = 'u'
    obser = np.array(ytest[i,0, :, :, 0])
    predit = np.array(testPredict[i+1, 0, :, :, 0])
    visualize(i,obser,predit,var)
    var = 'v'
    obser2 = np.array(ytest[i,0, :, :, 1])
    predit2 = np.array(testPredict[i+1, 0, :, :, 1])
    visualize(i,obser2,predit2,var)
pp.close()

# save model structure
model_json = model.to_json()
with open("modelthreelayer.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("modelthreelayer.h5")

'''



# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
'''


print("test")

print('Test Score: %.4f RMSE' % (testScore))
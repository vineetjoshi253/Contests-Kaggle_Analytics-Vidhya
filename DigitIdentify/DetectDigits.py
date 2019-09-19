import os
import csv
import numpy as np
import pandas as pd 
from PIL import Image
from matplotlib import pyplot
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

def ShowImage(TestImages,index):
    img = TestImages[index].reshape(28,28)
    pyplot.imshow(img)
    
def keepSort(TestImages):
    keepThis = []
    for item in TestImages:
        temp = int(item.replace('.png',''))
        keepThis.append(temp)
    
    keepThis.sort()
    TestImages = []
    for item in keepThis:
        TestImages.append(str(item)+'.png')
    return TestImages
        
    
def createTrainData():
    DATA_PATH=os.path.join(os.getcwd(),'Train_UQcUa52\Images\Train')
    TestImages = os.listdir(DATA_PATH)
    TestImages = keepSort(TestImages)
    TestData = []
    count = 1 
    for img_ittr in TestImages:
        IMAGE_PATH = os.path.join(DATA_PATH,img_ittr)
        img = Image.open(IMAGE_PATH).convert('L')
        img = np.asarray(img)
        img = img.reshape(28,28,1)
        TestData.append(img)
        print(count)
        count +=1
        
    TestData = np.asarray(TestData)    
    np.save('Training',TestData)

def InitModel(classes):
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(28, 28,1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


def createTestData():
    DATA_PATH=os.path.join(os.getcwd(),'Train_UQcUa52\Images')
    DATA_PATH = os.path.join(DATA_PATH,'test')
    TestImages = os.listdir(DATA_PATH)
    TestImages = keepSort(TestImages)
    TestData = []
    count = 1 
    for img_ittr in TestImages:
        IMAGE_PATH = os.path.join(DATA_PATH,img_ittr)
        img = Image.open(IMAGE_PATH).convert('L')
        img = np.asarray(img)
        img = img.reshape(28,28,1)
        TestData.append(img)
        print(count)
        count +=1
        
    TestData = np.asarray(TestData)    
    np.save('TestingData',TestData)
    
def createTrainLabels():
    DATA_PATH=os.path.join(os.getcwd(),'Train_UQcUa52\Images')
    DATA_PATH = os.path.join(DATA_PATH,'train.csv')
    labelData = pd.read_csv(DATA_PATH)        
    labels = np.asarray(labelData['label'])
    labels = np_utils.to_categorical(labels)
    np.save('TrainLabels',labels)
    
def getSolutions(model,TestingImages):
    with open('Solution.csv', mode='w',newline='') as myData:
        writer = csv.writer(myData)
        for index in range(len(TestingImages)): 
            pr = model.predict_classes(TestingImages[index].reshape(1,28,28,1))
            filename = index + 49000
            filename = str(filename)+'.png'
            writer.writerow([filename,pr[0]])
    
def main():
    
    #Create Dataset From Images
    #createTrainData()
    
    #Load Saved Training Dataset
    TrainingImages = np.load('Training.npy')
    
    #Normalize The Data
    TrainingImages = TrainingImages / 255
    
    #Create Label File
    #createTrainLabels
    
    #Load Labels
    labels = np.load('TrainLabels')
    
    #Create Test Set From Images
    #createTestData()
    
    #Load Test Dataset
    TestingImages = np.load('TestingData')
    
    
    classes = labels.shape[1]
    #Initialize The Model
    model = InitModel(classes)
    
    #Fit The Data
    model.fit(TrainingImages, labels,epochs=100, batch_size=200)
    
    #Accuracy On Training Data
    scores = model.evaluate(TrainingImages, labels, verbose=1)
    print("CNN Error: %.2f%%" % (100-scores[1]*100))
    
    TestingImages = np.load('TestingData.npy')
    
    getSolutions(model,TestingImages)
    
if __name__ == "__main__":
    main()
    



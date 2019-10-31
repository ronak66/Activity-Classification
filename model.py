import os
import numpy as np
import pandas as pd
import shutil

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

class Model:


    def __init__(self,file_path):
        (X,Y) = self.get_data(file_path)
        self.kerras_classifier(X,Y)
        return
        

    def get_data(self,file_path):
        df = pd.read_csv(file_path)

        dataset = df.values

        X = dataset[:,0:20].astype(float) # sensor data
        Y = dataset[:,20].astype(int) # labels
        return (X,Y)

    def create_model(self):
        # Define model
        model = Sequential()
        model.add(Dense(22, input_dim=20, activation='relu'))
        model.add(Dense(22, activation='relu'))
        model.add(Dense(22, activation='relu'))
        model.add(Dense(4, activation='softmax'))

        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def kerras_classifier(self,X_train,Y_train):
        self.model = KerasClassifier(self.create_model, epochs=200, batch_size=100, verbose=False)
        self.model.fit(X_train, Y_train)
        return self.model

    def result(self,X_test,Y_test):
        kfold = KFold(n_splits=10, shuffle=True, random_state=5)
        cv_results = cross_val_score(self.model, X_test, Y_test, cv=kfold)
        print("Baseline on test data: %.2f%% (%.2f%%)" % (cv_results.mean()*100, cv_results.std()*100))
        return cv_results

    def predict(self,X):
        return self.model.predict(X)


class activity_classifier:

    def __init__(self):
        self.acc_x_model = Model('Data/new/acc_X.csv')
        self.acc_y_model = Model('Data/new/acc_Y.csv')
        self.acc_z_model = Model('Data/new/acc_Z.csv')
        self.gyro_x_model = Model('Data/new/gyro_X.csv')
        self.gyro_y_model = Model('Data/new/gyro_Y.csv')
        self.gyro_z_model = Model('Data/new/gyro_Z.csv')
        return

    def accuracy(self):
        df_acc = pd.read_csv('Data/new/Accelerometer.csv')
        df_gyro = pd.read_csv('Data/new/Gyroscope.csv')
        count=0
        total=0
        for i in range(0,len(df_acc['X']),20):
            if(len(df_acc['X'][i:i+20].values)==20):
                p_acc_x = self.acc_x_model.predict(np.asarray([df_acc['X'][i:i+20].values]))
                p_acc_y = self.acc_y_model.predict(np.asarray([df_acc['Y'][i:i+20].values]))
                p_acc_z = self.acc_z_model.predict(np.asarray([df_acc['Z'][i:i+20].values]))
                p_gyro_x = self.gyro_x_model.predict(np.asarray([df_gyro['X'][i:i+20].values]))
                p_gyro_y = self.gyro_y_model.predict(np.asarray([df_gyro['Y'][i:i+20].values]))
                p_gyro_z = self.gyro_z_model.predict(np.asarray([df_gyro['Z'][i:i+20].values]))
                prediction = self.mode([p_acc_x[0],p_acc_y[0],p_acc_z[0],p_gyro_x[0],p_gyro_y[0],p_gyro_z[0]])
                if(df_acc['y'][i] == prediction):
                    count+=1
                total+=1
        return (count/total)*100

    def mode(self,array):
        return max(set(array), key = array.count) 
        # np.bincount(array).argmax()

if __name__ == '__main__':
    models = activity_classifier()
    print(models.accuracy())


    
import pandas as pd
import numpy as np

class Data_handling:

    def __init__(self,file_path):
        self.df = pd.read_csv(file_path)

    def split_dataset_to_X_Y_Z(self,save_path,file_type):
        df_name = pd.DataFrame(columns=range(21))
        for col in ['X','Y','Z']:
            k=0
            for i in range(0,len(self.df[col]),20):
                if(len(self.df[col][i:i+20].values)==20): 
                    df_name.loc[k] = np.append(self.df[col][i:i+20].values, self.df['y'][i])
                    k+=1
            df_name.to_csv(save_path+file_type+'_'+col+'.csv',index=False)
        return

    @staticmethod
    def merge_csv():
        for col in ['Accelerometer.csv','Gyroscope.csv']:
            df_climb = pd.read_csv('Data/new/climbing_0/'+col)
            df_jump = pd.read_csv('Data/new/jumping_1/'+col)
            df_run = pd.read_csv('Data/new/running_2/'+col)
            df_walk = pd.read_csv('Data/new/walking_3/'+col)
            df = df_climb.append(df_jump.append(df_run.append(df_walk,ignore_index = True),ignore_index = True),ignore_index = True)
            df.to_csv('Data/new/'+col,index=False)
        return

if __name__ == '__main__':

    Data_handling.merge_csv()

    df_acc = Data_handling('Data/new/Accelerometer.csv')
    df_acc.split_dataset_to_X_Y_Z('Data/new/','acc')
    df_gyro = Data_handling('Data/new/Gyroscope.csv')
    df_gyro.split_dataset_to_X_Y_Z('Data/new/','gyro')

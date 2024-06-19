import os
import numpy as np
import pandas as pd


###########################3
def mean_data(data,all_names):
    data_mean = []
    for i in range(len(data)):
        meandata = np.mean(data[i], axis=0) 
        data_mean.append(meandata)

    data_mean = np.array(data_mean)

    # dataframe
    df = pd.DataFrame(data_mean)
    df['video'] = all_names

    return df

def load_feature_data(direccion_video,direccion_audio=None, type_merge = None):

    # data video
    # Obtener el directorio de trabajo actual
    current_dir = os.getcwd()

    # Definir el subdirectorio de características relativo al directorio de trabajo actual
    feature_dir = os.path.join(current_dir, direccion_video)
    # Listar todos los archivos .npy en el subdirectorio
    feature_files = [f for f in os.listdir(feature_dir) if f.endswith('.npy')]
    # Cargar y procesar cada archivo .npy
    all_data_video = []
    all_names_video= []
    for file_name in feature_files:
        file_path = os.path.join(feature_dir, file_name)
        data = np.load(file_path)
        if data.shape == (0,):
            print(f'Skipping {file_name} because it is empty.')
        elif data.shape[1] == 512:
            print(f'Loaded {file_name[0:11]} with shape: {data.shape}')
            all_data_video.append(data)
            all_names_video.append(file_name[0:11])
        else:
            print(f'Skipping {file_name} due to incorrect shape: {data.shape}')

    print(f'Loaded {len(all_data_video)} files.')

    df_video = mean_data(all_data_video,all_names_video)

    
    if direccion_audio == None:
        data_final = df_video.drop(columns=['video']).to_numpy()
        print("data_final",data_final.shape)
        return data_final,df_video
    
    # data sound
    # Definir el subdirectorio de características relativo al directorio de trabajo actual
    feature_dir = os.path.join(current_dir, direccion_audio)

    # Listar todos los archivos .npy en el subdirectorio
    feature_files = [f for f in os.listdir(feature_dir) if f.endswith('.npy')]


    # Cargar y procesar cada archivo .npy
    all_data_sound = []
    all_names_sound= []
    for file_name in feature_files:
        file_path = os.path.join(feature_dir, file_name)
        data = np.load(file_path)
        if data.shape == (0,):
            print(f'Skipping {file_name} because it is empty.')
        elif data.shape[1] == 128:
            print(f'Loaded {file_name[0:11]} with shape: {data.shape}')
            all_data_sound.append(data)
            all_names_sound.append(file_name[0:11])
        else:
            print(f'Skipping {file_name} due to incorrect shape: {data.shape}')

    print(f'Loaded {len(all_data_sound)} files.')

    df_sound = mean_data(all_data_sound,all_names_sound)
    # uniendo dataframes
    if type_merge == 'left':
        df_final = pd.merge(df_video, df_sound, on='video', how='left')
        # relenar los valores nan con 0
        df_final.fillna(0, inplace=True)
        data_final = df_final.drop(columns=['video']).to_numpy()
        print("data_final",data_final.shape)  
    else:
        df_final = pd.merge(df_video, df_sound, on='video')
        data_final = df_final.drop(columns=['video']).to_numpy()
        print("data_final",data_final.shape)
    
    return data_final,df_final
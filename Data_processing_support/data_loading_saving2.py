import os
import numpy as np
import pandas as pd

def load_feature_data(feature_dir , format='.npy'):
    # Listar todos los archivos .npy en el directorio
    feature_files = [f for f in os.listdir(feature_dir) if f.endswith(format)]

    # Cargar y procesar cada archivo .npy
    all_data = []
    all_names = []
    for file_name in feature_files:
        file_path = os.path.join(feature_dir, file_name)
        data = np.load(file_path)
        if data.shape == (0,):
            print(f'Skipping {file_name} because it is empty.')
        elif data.shape[1] == 512:
            print(f'Loaded {file_name} with shape: {data.shape}')
            all_data.append(data)
            all_names.append(file_name[0:11])
        else:
            print(f'Skipping {file_name} due to incorrect shape: {data.shape}')

    print(f'Loaded {len(all_data)} files.')
    return all_data, all_names


def load_feature_data_sound(feature_dir , format='.npy'):
 
    feature_files = [f for f in os.listdir(feature_dir) if f.endswith(format)]

    # Inicializar listas para almacenar los datos y nombres de los archivos
    all_data_sound = []
    all_names_sound = []

    # Cargar y procesar cada archivo .csv
    for file_name in feature_files:
        file_path = os.path.join(feature_dir, file_name)
        try:
            if format=='.csv':
              data = pd.read_csv(file_path)
              if data.empty:
                    print(f'Skipping {file_name} because it is empty.')
              elif data.shape[1] == 128:
                    print(f'Loaded {file_name[0:11]} with shape: {data.shape}')
                    all_data_sound.append(data)
                    all_names_sound.append(file_name[0:11])
              else:
                    print(f'Skipping {file_name} due to incorrect shape: {data.shape}') 
                    
            else:
              data = np.load(file_path)
              if data.shape == (0,):
                    print(f'Skipping {file_name} because it is empty.')
              elif data.shape[1] == 128:
                    print(f'Loaded {file_name[0:11]} with shape: {data.shape}')
                    all_data_sound.append(data)
                    all_names_sound.append(file_name[0:11])
              else:
                    print(f'Skipping {file_name} due to incorrect shape: {data.shape}')   

         
        except pd.errors.EmptyDataError:
            print(f'Skipping {file_name} because it is empty or unreadable.')
        except Exception as e:
            print(f'Error loading {file_name}: {e}')

    print(f'Loaded {len(all_data_sound)} files.')
    return all_data_sound , all_names_sound

from pandas import read_excel
import csv
import pandas as pd
import numpy as np

def load_csv(filename):
    file = open(filename, "r", encoding="utf-8")
    experiments = []
    headers = []
    reader = csv.reader(file)
    for i, lines in enumerate(reader):
        if i == 0:
            #headers = [e for e in lines[0].split(";")]
            h = lines[0].split(";")
            headers.append(h)
        else:
            tmp = lines[0].split(";")
            for j in range(0,len(tmp)):
                if (j != 0 and j != 1879):
                    tmp[j] = float(tmp[j])
                else:
                    continue
            #tmp = [float(tmp[j]) for j in range(0,len(tmp)) if (j!=0 and j!=1879)] #This removes two entries, dunno why. 
            experiments.append(tmp)
    file.close()
    return headers, experiments

def column_extraction(data_list,column_number): #columns start counting in 0
    col = []
    for i in range(0,len(experiments)):
        dato = experiments[i][column_number]
        col.append(dato)
    return col 

def expetiment_to_library(library, experiment):
    notfound_count = 0
    notfound_index_list = []
    notfound_label_list = []
    notfound_name_list = []
    for index, row in experiment.iterrows():
        if index < 793: #Skip controls 
            continue
        else:
            compound = row['ID_covid']
            aux = library['ID_library'].loc[library['ID_library'] == compound].tolist()
            if len(aux) == 0:
                notfound_count += 1
                notfound_index_list.append(index)
                label = row['Labels']
                notfound_label_list.append(label)
                notfound_name_list.append(compound)
    data = {'Index': notfound_index_list,
            'Name': notfound_name_list,
            'Label': notfound_label_list}
    df = pd.DataFrame(data)
    print(f'There are {notfound_count} compounds not founded in SPECS')
    return df

def library_to_experiment(experiment, library): #Here you whant to know if all the elements in the library are also in the experiment
    notfound_count = 0
    notfound_index_list = []
    notfound_name_list = []
    for index, row in library.iterrows():
        compound = row['ID_library']
        aux = experiment['ID_covid'].loc[experiment['ID_covid'] == compound].tolist()
        if len(aux) == 0:
            notfound_count += 1
            notfound_index_list.append(index)
            notfound_name_list.append(compound)
    data = {'Index in library': notfound_index_list,
            'Name': notfound_name_list}
    df = pd.DataFrame(data)
    print(f'There are {notfound_count} compounds not founded in COVID')
    return df


def labelling(data, labels_position, label_positive, label_negative, threshold):
    labels = []
    for index, row in data.iterrows():
        mito_value = row['Intensity_MeanIntensity_illumMITO_cells'][1]
        if mito_value <= threshold:
            label = label_positive
        else:
            label = label_negative
        labels.append(label)
    data.insert(labels_position, "Labels", labels, True)
    return data

def select_median(experiments, df_to_be_ready, repeated_compounds):
    counter = 0
    median_repeated_compounds_idx = []
    in_size = len(df_to_be_ready) 
    
    for compound in repeated_compounds:
                
        np.random.seed(17)

        aux = experiments['Intensity_MeanIntensity_illumMITO_cells'].loc[experiments['CompoundID'] == compound] # Accessing info from the repeated compounds found in SPECS
        aux.columns = ['0','1'] #Renaming columns bcs of problem with column replication
        aux.drop('0', axis=1, inplace=True)
        aux2 = aux.sort_values(by=['1'], ascending=True) # Sort to access the median
        
        #if (counter > 42 & counter < 45):
        #    print(f'Compound {compound}')
        #    print(f' Auxiliar {aux2}')
        
        # Non-permanent solution
        if len(aux2) == 2:
            median_index = np.random.choice(aux.index.tolist())
            median = aux2['1'][median_index]
            if (counter > 42 & counter < 45):
                print(f'Median index {median_index}')
                print(median)
        else:
            median = aux2.median(axis=0).tolist()
            median_index = aux2.index[aux2['1'] == median[0]].tolist()
            median_index = median_index[0]

        row = experiments.iloc[[median_index]]
        median_repeated_compounds_idx.append(median_index)
        df_to_be_ready = df_to_be_ready.append(row)
        counter += 1

        if counter % 10 == 0:
            print(f'There are {len(df_to_be_ready)} rows in Covid Batch A')

    if len(repeated_compounds) +  in_size == len(df_to_be_ready):
        print('Congratulations! The file is ready')
    #Reordering indexes
    df_to_be_ready.reset_index(drop=True, inplace=True)
    
    return df_to_be_ready

def select_median_v2(experiments, df_to_be_ready, repeated_compounds):
    
    for i in range(len(repeated_compounds)):

        compound = repeated_compounds[i]

        np.random.seed(17)

        #print(f'Compound {compound} ') 

        #Find those compounds in the whole sss dataset
        indexes = experiments.index[experiments['CompoundID'] == compound].tolist()
        temp1 = experiments.iloc[indexes, [-1]]
        #print(f'List of indices: {indexes}')
        if len(temp1) == 2:
            median_index = np.random.choice(indexes)
            #print(median_index)
        else:
            median = temp1.median().tolist()[0]
            median_index = temp1.index[temp1['Intensity_MeanIntensity_illumMITO_cells'] == median].tolist()[0]
            #print(median_index)

        row = experiments.iloc[[median_index]]
        #print(f'Row info {row.info()}')

        df_to_be_ready = df_to_be_ready.append(row)

        ids = df_to_be_ready["CompoundID"]
        temp2 = df_to_be_ready[ids.isin(ids[ids.duplicated()])]
        if len(temp2) > 0:
            print("Repeticioooooooon")
            break    

    #print(f'Finally done \n The lenght now is {len(df_to_be_ready)} rows')
    #Reordering indexes
    df_to_be_ready.reset_index(drop=True, inplace=True)
    
    return df_to_be_ready


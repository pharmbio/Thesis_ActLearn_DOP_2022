from pandas import read_excel
import csv
import pandas as pd

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

def sss_vs_specs_compound_detection(library, experiment):
    notfound_count = 0
    notfound_index_list = []
    notfound_label_list = []
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
    print(f'There are {notfound_count} compounds not founded')
    return notfound_index_list , notfound_label_list

def labelling(data, labels_position, label_positive, label_negative, threshold):
    labels = []
    for index, row in data.iterrows():
        mito_value = row['Intensity_MeanIntensity_illumMITO_cells.1']
        if mito_value <= threshold:
            label = label_positive
        else:
            label = label_negative
        labels.append(label)
    data.insert(labels_position, "Labels", labels, True)
    return data
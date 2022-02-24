from pandas import read_excel
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import random

from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from modAL.models import ActiveLearner
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cross_decomposition import PLSRegression

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem,DataStructs,Draw,PandasTools,Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Draw import IPythonConsole
from rdkit import RDLogger

from IPython.display import Image

# Configure the logging - RDKit is rather verbose..
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
# Set the molecule representation to be SVG 
PandasTools.molRepresentation='svg'
# Loading and standardization method for SMILES -> RDKit molecule object
uncharger = rdMolStandardize.Uncharger()


# Loading and standardization method for SMILES -> RDKit molecule object
uncharger = rdMolStandardize.Uncharger()


# ********************************************************************************
# *************************** FUNCTIONS ******************************************
# ********************************************************************************


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

def standardize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mol = rdMolStandardize.FragmentParent(mol)
        mol = uncharger.uncharge(mol)
    return mol

# split dataset into test set, train set and unlabel pool
def split(x_dataset, y_dataset, ini_train_size, test_size):
    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size = test_size, random_state=23654,shuffle =True)
    x_labelled, x_pool, y_labelled, y_pool = train_test_split(x_train, y_train, train_size = ini_train_size,random_state=23654, shuffle = True)
    return x_labelled, y_labelled, x_test, y_test, x_pool, y_pool 

def plot_incremental_accuracy(performance_history, save, figure_name):
# Plot our performance over time.
    fig, ax = plt.subplots(figsize=(6,4), dpi=130)

    #ax.axhline(y=.95, xmin=0, xmax=20, color='r', linestyle='-', linewidth=1.5)
    #ax.plot(performance_history, c = )
    ax.scatter(range(len(performance_history)), performance_history, s=15, 
               edgecolor=(.937, .275,.282), linewidth=0.1, facecolor=(.937, .275, .282))

    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
    ax.set_ylim(bottom=0.95, top=1)
    #ax.axes.autoscale(enable=True, axis='y', tight=True)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    ax.grid(True)

    ax.set_title('Incremental classification accuracy')
    ax.set_xlabel('Query iteration')
    ax.set_ylabel('Classification Accuracy')
    if save:
        plt.savefig("".join(["incr_accu_", figure_name, ".jpg"]), bbox_inches='tight')
    #plt.show()
    
    
def feature_creation(morgan_radius, morgan_n_bits, fp_n_bits, data):
    # generate Morgan fingerprint with radius 2
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, morgan_radius, nBits=morgan_n_bits) for m in data['MOL']]
    # convert the RDKit explicit vectors into numpy arrays
    X_morgan = np.asarray(fps)

    # generate RDKFingerprint with default settings
    rdkit_fp = [Chem.RDKFingerprint(m, fpSize=fp_n_bits) for m in data['MOL']]
    # convert the RDKit explicit vectors into numpy arrays
    X_rdkit = np.asarray(rdkit_fp)

    # Get the target values 
    y = data['Labels'].to_numpy()
    
    return X_morgan, X_rdkit, y

def active_learnig_train(n_queries, x_train, y_train, x_test, y_test, x_pool, y_pool, Classifier):
    
    performance_history = []
    cf_matrix_history = []
    learner = ActiveLearner(estimator=Classifier, X_training = x_train, y_training = y_train)
    
    #Making predictions
    y_pred = learner.predict(x_test)

    #Calculate and report our model's accuracy.
    model_accuracy = learner.score(x_test, y_test)
    
    #Generate the confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    
    # Save our model's performance for plotting.
    performance_history.append(model_accuracy)
    cf_matrix_history.append(cf_matrix)
    
    # Allow our model to query our unlabeled dataset for the most
    # informative points according to our query strategy (uncertainty sampling).
    for index in range(n_queries):
        
        #Query for a new point
        query_index, query_instance = learner.query(x_pool)

        # Teach our ActiveLearner model the record it has requested.
        XX, yy = x_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
        learner.teach(X=XX, y=yy)

        # Remove the queried instance from the unlabeled pool.
        x_pool, y_pool = np.delete(x_pool, query_index, axis=0), np.delete(y_pool, query_index)
        
        y_pred = learner.predict(x_test)
        model_accuracy = learner.score(x_test, y_test)
        cf_matrix = confusion_matrix(y_test, y_pred)
        performance_history.append(model_accuracy)
        cf_matrix_history.append(cf_matrix)

      
        if index % 30 == 0:
            print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))
        
    return performance_history , cf_matrix_history

def plot_cf_mat(matrix, save, figure_name):
    fig, ax = plt.subplots(figsize=(5,4), dpi=130)
    ax = sns.heatmap(matrix/np.sum(matrix), annot=True, fmt = '.2%', cmap=sns.light_palette((.376, .051, .224)))
    ax.set_title('Confusion Matrix\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])
    
    if save:
        plt.savefig("".join(["cf_mat_", figure_name, ".jpg"]), bbox_inches='tight')

    ## Display the visualization of the Confusion Matrix.
    #plt.show()
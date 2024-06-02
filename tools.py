import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import pandas as pd
from math import ceil
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

### Bouts de code utiles ###
#import importlib
#import tools
#importlib.reload(tools)
#ax.set_xlim(data.num3.min(),data.num3.max())
#plt.yscale("log")
#data.groupby(["x","y"]).size().unstack(fill_value=0)
#%%timeit
#%matplotlib inline
#%matplotlib notebook

### OUTILS POUR PLOTS ###

def labelsizes(ax):
    ax.yaxis.label.set_size(18)
    ax.xaxis.label.set_size(18)
    ax.title.set_size(20)

def adjustfig(fig):
    fig.tight_layout()
    fig.subplots_adjust(top=0.98,right=0.95,left=0.05,bottom=0.02)

def adjustfig2(fig):
    fig.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.16, 
                    hspace=0.2)
    
def auto_subplots(nplots,ncols,rectangle=False, length=20, title=False,height_coef=1):
    if nplots == 1: nrows, ncols, height, length = 1, 1, 5, 5
    else:
        nrows = ceil(nplots/ncols)
        coef = 8.5 if nrows == 1 else 9
        if rectangle:
            height = 1.7 * nrows * ncols
        else:
            height = 2 * coef * nrows / ncols #Avant : sans le 9 et multiplié par ncols
    

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(length,height))
    if (ncols > 1) and (nrows > 1): axes = axes.flatten()
    return fig, axes, height
    
    
### OUTILS DE STAT / EXPLORATION SANS GRAPHIQUES (PRINT / TABLEAUX) ###

def value_percents(df,col):
    value_counts = df[col].value_counts()
    temp = np.round(value_counts.values / df.shape[0],3)
    return pd.Series(temp,index=value_counts.index)
    

    
def split_df(bftgh1,ngroups):
    max_smiss = bftgh1.smiss.max()
    step = ceil((max_smiss+1) / ngroups)
    bftgh1s = []
    i = 0
    while i < max_smiss:
        prev_i = i
        i+= step
        #print(prev_i,i)
        new = bftgh1[(bftgh1.smiss >= prev_i) & (bftgh1.smiss < i)]
        if not new.empty: bftgh1s.append(new)
    return bftgh1s
    
def get_missing(data):
    percent_missing = data.isnull().sum() * 100 / len(data)
    missing_value_df = pd.DataFrame({'Nom de colonne': data.columns,
                                    'Pourcentage de NA': percent_missing}).sort_values(ascending=True, by="Pourcentage de NA")
    return missing_value_df

def print_nrows(data):
    print("Nombre de lignes : ", data.shape[0])
    
def headtail(data, n=5):
    return pd.concat([data.head(n),data.tail(n)])

from random import randint
def explore(df,n=5):
    nrows = range(df.shape[0])
    ix = randint(nrows.start, nrows.stop-n)
    return df.iloc[ix:ix+n, :]

def find_inf(temp):
    for col in temp.columns:
        summ = np.isinf(temp[col]).sum()
        if summ > 0:
            print(col,summ)
            
def filter_outliers(data, columns, i=1.5):
    info_list = []
    limited_data = data
    for col in columns:
        q1 = data[col].quantile(q=0.25)
        q3 = data[col].quantile(q=0.75)
        iqr = q3-q1
        lim_inf = q1 - i * iqr
        lim_sup = q3 + i * iqr
        info_list.append([col,lim_inf,lim_sup,data[(data[col] < lim_inf) | (data[col] > lim_sup)].shape[0]])
        limited_data = limited_data.loc[~((limited_data[col] < lim_inf) | (limited_data[col] > lim_sup))]
    return limited_data, pd.DataFrame(info_list, columns=["Coordonnée","Seuil inférieur","Seuil supérieur","Nombre d'outliers"])

def split_cols(data,fake_num_cols=None):
    #On crée une liste de colonnes numériques et une liste de colonnes catégoriques (ordinal + catégorie)
    cat_col = list(data.select_dtypes('object').columns)
    num_col = list(data.select_dtypes(np.number).columns)
    if fake_num_cols is not None:
        num_col = [x for x in num_col if x not in fake_num_cols]
        cat_col += fake_num_cols
    print("Colonnes continues: ", num_col)
    print("Colonnes discrètes: ", cat_col)
    return num_col, cat_col


def get_timestamp_column(data, ref_date = '2023-05-28', date_column="date", datetime_as_index=False):
    reference_timestamp = pd.Timestamp('2023-05-28').timestamp()
    if datetime_as_index:
        return data.index.values.astype(np.float64) - reference_timestamp
    else:
        return data[date_column].values.astype(np.float64) - reference_timestamp
    
def print_value_counts(data,columns, n_max=20):
    for column in columns:
        print(data[column].value_counts())
    
def gtia(df,trig2):
    return df[df.trig2 == trig2].iloc[0].gtia

def print_ratio(data,cond,msg):
    print_ratio2(data,data[cond],msg)
    
def print_ratio2(data,data2,msg,inv=False):
    print_ratio3(data.shape[0],data2.shape[0],msg,inv=inv)
    
def print_ratio3(data_nrows,data2_nrows,msg,inv=False,n_decimals=2):
    if inv:
        data2_nrows = (data_nrows - data2_nrows)
    ratio = round((data2_nrows / data_nrows)*100,n_decimals)
    if n_decimals==0: ratio = int(ratio)
    print(f"{ratio}% de {msg} ({data2_nrows} sur {data_nrows})")
    
def printratio(data,col,v=True):
    print(f"{round((data[data[col] == v].shape[0] / data.shape[0])*100,2)} % de {col} == {v} sur {data.shape[0]} lignes")
    
   
### OUTILS GENERAUX DE VISUALISATION ###

def plotcorrmatrix(data, figsize=20):
    corrMatrix = data.corr()
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(figsize,figsize))
    ax = sns.heatmap(corrMatrix, annot=True, cmap="Blues")
    plt.show()

def cat_plot(data, col, ystr, type, title=None, hue=None,debug=False):
    """
    input : df, liste des variables discrètes (list(str)), target (str)
    action : trace graphiques pertinents de chaque variable discrète par rapport à target
    output : figure matplotlib
    """
    plt.rc('legend', fontsize = 14)
    col = [x for x in col if (len(data[x].unique()) < 30) & (col != ystr)]
    fig, axes = plt.subplots(nrows = ceil(len(col)/2), ncols=2, figsize=(20,4 * len(col)))
    axs = axes.flatten()
    j = 0
    if type == "regression":
        if title == None:
            fig.suptitle('Distribution of ' + ystr + ' with respect to categories', fontsize=22)
        for i in col:
            if debug: print(i)
            ax = sns.boxplot(data=data, x=i, y=ystr, ax=axs[j])
            labelsizes(ax)
            if len(data[i].unique()) > 5:
                ax.tick_params(labelrotation=90)
            j = j + 1
    elif type == "classification":
        if title is None:
            fig.suptitle('Probability of each class of ' + ystr + ' for all categories', fontsize=22)
        for i in col:
            if debug: print(i)
            ax = sns.histplot(data=data, x=i, hue=ystr, ax=axs[j], multiple="stack", stat="probability")
            labelsizes(ax)
            if len(data[i].unique()) > 5:
                ax.tick_params(labelrotation=90)
            j = j + 1
    if len(col) % 2 != 0:
        axs[len(col)].set_axis_off()
    adjustfig(fig)
    if title is not None:
        fig.suptitle(title, fontsize=21, y=1.05)
    plt.show()

def num_plot(data, col, ystr, type, kde=True, title=None, hue=None, facetgrid=False, boxplot=False,debug=False, **kwargs):
    """
    input : df, liste des variables continues (list(str)), target (str)
    action : trace graphiques pertinents de chaque variable continue par rapport à target
    output : figure matplotlib
    """
    plt.rc('legend', fontsize = 14)
    fig, axes = plt.subplots(nrows = ceil(len(col)/2), ncols=2, figsize=(20,4 * len(col)))
    axs = axes.flatten()
    j = 0
    if type == "regression":
        if title == None:
            title = 'Scatter plot de ' + ystr + ' en fonction des variables numériques (+ line of best fit)'
        for i in col:
            if debug: print(i)
            if i != ystr:
                if hue != None:
                    ax = sns.lineplot(data=data, x=ystr, y=i, ax=axs[j], hue=hue)
                else:
                    ax = sns.regplot(data=data, x=i, y=ystr, ax=axs[j], scatter_kws={"s": 2}, line_kws={'color': 'red'})
                labelsizes(ax)
                j = j + 1
    elif type == "classification":
        if debug: print(i)
        if title == None:
            title = 'Evolution de la densité de chaque classe ' + ystr + ' en fonction des variables numériques'
        for i in col:
            if i != ystr:
                if (kde):
                    ax = sns.kdeplot(data=data, x=i, hue=ystr, ax=axs[j], **kwargs)
                else:
                    if (boxplot):
                        ax = sns.boxplot(data=data,x=ystr,y=i,ax=axs[j])
                    else:
                        ax = sns.histplot(data=data, x=i, hue=ystr, ax=axs[j], element="poly",stat="probability", fill=False)
                labelsizes(ax)
                if (boxplot) and (len(data[i].unique()) > 5):
                    ax.tick_params(labelrotation=90)
                j = j+1
    if len(col) % 2 != 0:
        axs[len(col)].set_axis_off()
    fig.suptitle(title, fontsize=21, y=1.05)
    adjustfig(fig)
    plt.show()

def y_distribution(data, col):
    """
    input : df, colonne
    action : trace distribution de la colonne
    output : figure matplotlib
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6))
    ax = sns.histplot(data=data, x=col, ax=ax, stat="probability")
    plt.show()
    
def distribution(data, col,debug=False, title='Distribution des variables', y=1.05, hue=None, stat="probability"):
    """
    input : df, liste de colonnes
    action : trace distribution de toutes les colonnes
    output : figure matplotlib
    """
    fig, axes = plt.subplots(nrows = ceil(len(col)/2), ncols=2, figsize=(20,4 * len(col)))
    axs = axes.flatten()
    j = 0
    for i in col:
        if debug: print(i)
        #max_n = 30
        #if (len(data[i].unique()) > 30):
        #top_categories = data[i].value_counts().sort_values(ascending=False)[:
        ax = sns.histplot(data=data, x=i, ax=axs[j], hue=hue)
        labelsizes(ax)
        if (len(data[i].unique()) > 5):
            ax.tick_params(labelrotation=90)
        j = j + 1
    if len(col) % 2 != 0:
        axs[len(col)].set_axis_off()
    adjustfig(fig)
    fig.suptitle(title, fontsize=20, y=y)
    plt.show()

def plot_nunique_values(data):
    unique_values = data.nunique().sort_values()
    unique_values.plot.bar(logy=True, figsize=(15, 4), title="Valeurs uniques par colonne")

def get_unique_values(data,cat_cols):
    for i in cat_cols:
        print(i,data[i].unique(),sep=':\n',end='\n\n')
        
def perform_pca(data,col,normalize=True):
    pca = PCA(2)
    if (normalize):
        scaler = StandardScaler() 
        data = scaler.fit_transform(data)
    score = pca.fit_transform(data)
    #On affiche les variances de chaque PC
    exp_var_pca = pca.explained_variance_ratio_
    print("Variance expliquée par le 1er composant : ", exp_var_pca[0])
    print("Variance expliquée par le 2eme composant : ", exp_var_pca[1])
    #On affiche les coeffs pour chaque colonne
    pcadf = pd.DataFrame(pca.components_, columns=col, index=['PC1', 'PC2'])
    return pca, score, pca.components_, pcadf

def biplot(score,coef,labels=None):
    fig, ax = plt.subplots(figsize=(10,10))

    #Cercle de corrélation
    an = np.linspace(0, 2 * np.pi, 100)    
    plt.plot(np.cos(an), np.sin(an), c='g')

    #Paramètres
    ax.set_aspect('equal')
    ax.grid(True, which='both')
    sns.despine(ax=ax, offset=0)
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)

    #Scatter plot
    xs = score[:,0]
    ys = score[:,1]
    coef = np.transpose(coef)
    n = coef.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley,
                s=4, 
                color='b')
 
    #Loading vectors
    for i in range(n):
        plt.arrow(0, 0, coef[i,0], 
                  coef[i,1],color = 'red',
                  alpha = 0.5)
        plt.text(coef[i,0]* 1.15, 
                 coef[i,1] * 1.15, 
                 labels[i], 
                 color = 'r', 
                 ha = 'center', 
                 va = 'center')
 
    #Afficher
    plt.xlabel("PC{}".format(1), fontsize=16)
    plt.ylabel("PC{}".format(2), fontsize=16)    
    plt.figure()
    plt.show()

def plot_cat_facetgrid(data,cols,y,categ,i):
    g = sns.FacetGrid(data,row=y,col=categ)
    g.map_dataframe(sns.histplot, x=cols[i], stat='probability')
    if i > len(cols):
        print("Fini de plot toutes les colonnes")
    g.add_legend()
    g.fig.suptitle(cols[i] + " en fonction de la minute et du type de log", fontsize=22, y=0.95)
    adjustfig(g.fig)
    plt.show()




### INTERFACE (console) ###
class color:
    BOLD = '\033[1m'
    HEADER = '\033[95m'
    HEADER2 = '\033[95m'
    HEADER3 = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    SMALL_FAIL = '\033[91m'
    UNDERLINE = '\033[4m'
    ITALIC = '\033[0m'
    GREY = ''
    RESET = '\033[0m'
    
def show_message(msg,specific_color):
    print(specific_color+msg+color.RESET)
    
def show_error(msg,color_prefix=color.FAIL):
    show_message(msg,color_prefix)
    
def show_warning(msg,color_prefix=color.WARNING):
    show_message(msg,color_prefix)

def show_success(msg,color_prefix=color.OKGREEN):
    show_message(msg,color_prefix)
    
def show_fail(msg,color_prefix=color.SMALL_FAIL):
    show_message(msg,color_prefix)
    
def show_info(msg,color_prefix=''):
    show_message(msg,color_prefix)
    
def show_note(msg,color_prefix=color.ITALIC):
    show_message(msg,color_prefix)

def show_intro(msg,color_prefix=color.HEADER):
    show_message(msg,color_prefix)
    
def show_intro2(msg,color_prefix=color.HEADER2):
    show_message(msg,color_prefix)
    
def show_intro3(msg,color_prefix=color.HEADER3):
    show_message(msg,color_prefix)
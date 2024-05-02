"""
Projeto Final - Fault Analysis in Industrial Equipment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pandas.tseries.offsets import Minute
from statsmodels.tsa.stattools import adfuller
import seaborn as sns

dataLoading = pd.ExcelFile("C:/Users/Utilizador/Downloads/Material_Projeto/Material_Projeto/DadosProjeto/Dados 2019.xlsx") #Carregamento do ficheiro Excel

dataCleaned = [] #Variável que contêm uma lista de DataFrames com os dados processados

shortColumns = ["BA1", "BA2", "BA3",
                "VB1", "VB2", "VB3",
                "TOB1", "TOB2", "TOB3",
                "PB1", "PB2", "PB3",
                "CBA",
                "TEuBA1", "TEvBA1", "TEwBA1", "TRLABA1", "TRLCABA1",
                "TEuBA2", "TEvBA2", "TEwBA2", "TRLABA2", "TRLCABA2",
                "TEuBA3", "TEvBA3", "TEwBA3", "TRLABA3", "TRLCABA3",
                "VRBA1", "VRBA2", "VRBA3"]

dateRangeAnually2017 = [["2017-01-01 00:00:00", "2017-02-01 00:00:00"],
                    ["2017-02-01 00:01:00", "2017-02-28 23:59:00"],
                    ["2017-03-01 00:00:00", "2017-04-01 00:00:00"],
                    ["2017-04-01 00:01:00", "2017-05-01 00:00:00"],
                    ["2017-05-01 00:01:00", "2017-06-01 00:00:00"],
                    ["2017-06-01 00:01:00", "2017-07-01 00:00:00"],
                    ["2017-07-01 00:01:00", "2017-08-01 00:00:00"],
                    ["2017-08-01 00:01:00", "2017-09-01 00:00:00"],
                    ["2017-09-01 00:01:00", "2017-10-01 00:00:00"],
                    ["2017-10-01 00:01:00", "2017-11-01 00:00:00"],
                    ["2017-11-01 00:01:00", "2017-12-01 00:00:00"],
                    ["2017-12-01 00:01:00", "2018-01-01 00:00:00"]
                   ]

dateRangeAnually2018 = [["2018-01-01 00:01:00", "2018-02-01 00:00:00"],
                        ["2018-02-01 00:01:00", "2018-02-28 23:59:00"],
                        ["2018-03-01 00:00:00", "2018-04-01 00:00:00"],
                        ["2018-04-01 00:01:00", "2018-05-01 00:00:00"],
                        ["2018-05-01 00:01:00", "2018-06-01 00:00:00"],
                        ["2018-06-01 00:01:00", "2018-07-01 00:00:00"],
                        ["2018-07-01 00:01:00", "2018-08-01 00:00:00"],
                        ["2018-08-01 00:01:00", "2018-09-01 00:00:00"],
                        ["2018-09-01 00:01:00", "2018-10-01 00:00:00"],
                        ["2018-10-01 00:01:00", "2018-11-01 00:00:00"],
                        ["2018-11-01 00:01:00", "2018-12-01 00:00:00"],
                        ["2018-12-01 00:01:00", "2019-01-01 00:00:00"]
                       ]

dateRangeAnually2019 = [["2019-01-01 00:01:00", "2019-02-01 00:00:00"],
                        ["2019-02-01 00:01:00", "2019-02-28 23:59:00"],
                        ["2019-03-01 00:00:00", "2019-04-01 00:00:00"],
                        ["2019-04-01 00:01:00", "2019-05-01 00:00:00"],
                        ["2019-05-01 00:01:00", "2019-06-01 00:00:00"],
                        ["2019-06-01 00:01:00", "2019-07-01 00:00:00"],
                        ["2019-07-01 00:01:00", "2019-08-01 00:00:00"],
                        ["2019-08-01 00:01:00", "2019-09-01 00:00:00"],
                        ["2019-09-01 00:01:00", "2019-10-01 00:00:00"],
                        ["2019-10-01 00:01:00", "2019-11-01 00:00:00"],
                        ["2019-11-01 00:01:00", "2019-12-01 00:00:00"],
                        ["2019-12-01 00:01:00", "2019-12-31 23:59:00"]
                       ]

def processData(dataToClean, columnsNames, indexToUse):
    #Este método assume que não existem timestamps duplicados em cada folha de Excel e que os mesmos estão ordenados de forma crescente
    
    #dataToClean -> DataFrame que contém os valores dos dados a serem processados
    #columnsNames -> Nome das colunas associadas ao dataFrame a ser devolvido
    #indexToUse -> Index do dataFrame a ser devolvido
    
    dummyDf = pd.DataFrame(np.nan,
                           index = indexToUse,
                           columns = columnsNames,
                           dtype = "float64")

    columnsToAnalyze = dataToClean.columns.size // 2 #As colunas deverão ser analisadas aos pares
    numRows = len(indexToUse) #Número de linhas resultante dos períodos e frequência definidos aquando da criação do objeto DateTimeIndex
    numColumnToInsertValues = 0 #Número da coluna a inserir valores no dataFrame a ser utilizado para receber os dados

    #Condições iniciais dos indexes das colunas a retirar informação
    i = 0 #Variável i corresponderá aos timestamps das variáveis presentes nas diferentes colunas
    j = 1 #Variável j corresponderá aos valores das variáveis presentes nas diferentes colunas
    rowToProcess = 0 #Corresponde ao número da linha que contêm o timestamp a analisar

    #Condição inicial do index responsável por quantificar o número de iterações necessárias de forma a analisar o dataFrame de forma correta
    pairsOfColumnsProcessed = 0

    while pairsOfColumnsProcessed < columnsToAnalyze:    
        for numberRow in range(numRows):
            if dummyDf.index[numberRow] == dataToClean.iloc[rowToProcess,i]: #Comparação de dois timestamps
                dummyDf.iloc[numberRow, numColumnToInsertValues] = dataToClean.iloc[rowToProcess,j]
                rowToProcess += 1
            else:
                dummyDf.iloc[numberRow, numColumnToInsertValues] = np.nan
        
        i += 2
        j += 2
        rowToProcess = 0
        numColumnToInsertValues += 1
        pairsOfColumnsProcessed += 1 

    return dummyDf

for index in range(len(dataLoading.sheet_names)):
    dataToFormat = pd.read_excel(dataLoading, dataLoading.sheet_names[index], skiprows = [0,1,2,3,4,5])
    indexToUse = pd.date_range(start = dateRangeAnually2019[index][0], end = dateRangeAnually2019[index][1], freq = "T")
    dataCleaned.append(processData(dataToFormat, shortColumns, indexToUse))
    
data2017 = pd.concat(dataCleaned) #Concatenação dos dados processados num único objeto do tipo DataFrame

data2017_filled = data2017.fillna(-5)

data2017.loc["2019-08", ["VB1","VB2","VB3"]].describe() #Segmentação a nível mensal

data2017.loc["2019", ["VB1","VB2","VB3"]].describe()

data2017_filled.loc["2019", ["VB1","VB2","VB3"]].describe()

#Processo de criação de um gráfico com os dados e respetivos labels dos mesmos utilizando o matplotlib

fig, ax = plt.subplots()
ax.plot(data2017.loc["2019"]["VB1"], color = "blue", label = "VB1 (mm/s)")
ax.plot(data2017.loc["2019"]["VB2"], color = "orange", label = "VB2 (mm/s)")
ax.plot(data2017.loc["2019"]["VB3"], color = "green", label = "VB3 (mm/s)")
ax.legend()
ax.set(title = "Dados 2019 Vibração - Bomba de Aparas", xlabel = "Meses")
plt.show()

fig, ax = plt.subplots()
ax.plot(data2017_filled.loc["2019"]["VB1"], color = "blue", label = "VB1 (mm/s)")
ax.plot(data2017_filled.loc["2019"]["VB2"], color = "orange", label = "VB2 (mm/s)")
ax.plot(data2017_filled.loc["2019"]["VB3"], color = "green", label = "VB3 (mm/s)")
ax.legend()
ax.set(title = "Dados 2019 Vibração - Bomba de Aparas", xlabel = "Meses")
plt.show()

cols_with_missing_values = (data2017.loc["2019"].isnull().sum()) #Contagem de valores nulos presentes em cada coluna
cols_with_missing_values = (data2017.loc["2019"][["VB1", "VB2","VB3"]].isnull().sum()) #Contagem de valores nulos presentes nas colunas VB1, VB2 e VB3

#Processo de serialização e desserialização dos dados a serem analisados utilizando o pickle
filename = "C:/Users/Utilizador/Downloads/Material_Projeto/Material_Projeto/dataProcessed"
#pickle.dump(data2017, open(filename, "wb"))
data = pickle.load(open(filename, "rb"))

data.loc["2017"][["VB1", "VB2", "VB3"]].describe() #Segmentação anual das variáveis a serem analisadas
data.loc["2018"][["VB1", "VB2", "VB3"]].describe() #Segmentação anual das variáveis a serem analisadas
data.loc["2019"][["VB1", "VB2", "VB3"]].describe() #Segmentação anual das variáveis a serem analisadas

def processDataWeekly(data):
    indexWeekly = pd.date_range(start = "2017-01-01 00:00:00", end = "2019-12-31 23:59:00", freq = "W")
    size = len(indexWeekly) - 1
    minute = Minute(1)
    weeklyNanValues = []

    for i in range(size):
        startWeek = indexWeekly[i]
        endWeek = indexWeekly[i + 1] - minute
        labelOfWeek = " :: ".join([str(startWeek), str(endWeek)])
        
        vb1 = data.loc[startWeek:endWeek][["VB1"]].isnull().sum()["VB1"]
        vb2 = data.loc[startWeek:endWeek][["VB2"]].isnull().sum()["VB2"]
        vb3 = data.loc[startWeek:endWeek][["VB3"]].isnull().sum()["VB3"]
        values = {"VB1": vb1, "VB2": vb2, "VB3": vb3}

        df = pd.DataFrame(data = values,
                          index = [labelOfWeek],
                          columns = ["VB1", "VB2", "VB3"])
        
        weeklyNanValues.append(df)

    dfWeekly = pd.concat(weeklyNanValues)
    return dfWeekly

weeksProcessed = processDataWeekly(data)

# Segmentação dos dados a nível semanal analisando o número de valores omissos
weeksProcessed.loc["2017-01-01 00:00:00 :: 2017-01-07 23:59:00":"2017-12-24 00:00:00 :: 2017-12-30 23:59:00"] #Ano 2017
weeksProcessed.loc["2017-12-31 00:00:00 :: 2018-01-06 23:59:00":"2018-12-23 00:00:00 :: 2018-12-29 23:59:00"] #Ano 2018
weeksProcessed.loc["2018-12-30 00:00:00 :: 2019-01-05 23:59:00":"2019-12-22 00:00:00 :: 2019-12-28 23:59:00"] #Ano 2019

#Contagem do número de valores omissos usando como espaço temporal um trimestre
data.loc["2019-01-01 00:00:00":"2019-03-31 23:59:00"][["VB1", "VB2","VB3"]].isnull().sum() #1º trimestre
data.loc["2019-04-01 00:00:00":"2019-06-30 23:59:00"][["VB1", "VB2","VB3"]].isnull().sum() #2º trimestre
data.loc["2019-07-01 00:00:00":"2019-09-30 23:59:00"][["VB1", "VB2","VB3"]].isnull().sum() #3º trimestre
data.loc["2019-10-01 00:00:00":"2019-12-31 23:59:00"][["VB1", "VB2","VB3"]].isnull().sum() #4º trimestre

dataToResample = data.copy()
dataToResample = dataToResample.resample("5min", closed = "right", label = "right").mean() #Downsampling
#weeklyResample = data.resample("W-SUN", closed = "right", label = "right").mean()

dataToResample.loc["2017", ["VB1","VB2","VB3"]].describe()
dataToResample.loc["2018", ["VB1","VB2","VB3"]].describe()
dataToResample.loc["2019", ["VB1","VB2","VB3"]].describe()

weeksResampled = processDataWeekly(dataToResample)

#Segmentação dos dados a nível semanal analisando o número de valores omissos
weeksResampled.loc["2017-01-01 00:00:00 :: 2017-01-07 23:59:00":"2017-12-24 00:00:00 :: 2017-12-30 23:59:00"] #Ano 2017
weeksResampled.loc["2017-12-31 00:00:00 :: 2018-01-06 23:59:00":"2018-12-23 00:00:00 :: 2018-12-29 23:59:00"] #Ano 2018
weeksResampled.loc["2018-12-30 00:00:00 :: 2019-01-05 23:59:00":"2019-12-22 00:00:00 :: 2019-12-28 23:59:00"] #Ano 2019

#Contagem do número de valores omissos usando como espaço temporal um trimestre
dataToResample.loc["2019-01-01 00:00:00":"2019-03-31 23:59:00"][["VB1", "VB2","VB3"]].isnull().sum() #1º trimestre
dataToResample.loc["2019-04-01 00:00:00":"2019-06-30 23:59:00"][["VB1", "VB2","VB3"]].isnull().sum() #2º trimestre
dataToResample.loc["2019-07-01 00:00:00":"2019-09-30 23:59:00"][["VB1", "VB2","VB3"]].isnull().sum() #3º trimestre
dataToResample.loc["2019-10-01 00:00:00":"2019-12-31 23:59:00"][["VB1", "VB2","VB3"]].isnull().sum() #4º trimestre

#Escrita num ficheiro Excel
writer = pd.ExcelWriter("C:/Users/Utilizador/Desktop/ISEC - LEI/4º ANO/2º SEMESTRE/Projeto/Semanas2019Resampled.xlsx")
weeksResampled.loc["2018-12-30 00:00:00 :: 2019-01-05 23:59:00":"2019-12-22 00:00:00 :: 2019-12-28 23:59:00"].to_excel(writer, "2019")
writer.close()

Weeks2017 = weeksResampled.loc["2017-01-01 00:00:00 :: 2017-01-07 23:59:00":"2017-12-24 00:00:00 :: 2017-12-30 23:59:00"].copy()
Weeks2018 = weeksResampled.loc["2017-12-31 00:00:00 :: 2018-01-06 23:59:00":"2018-12-23 00:00:00 :: 2018-12-29 23:59:00"].copy()
Weeks2019 = weeksResampled.loc["2018-12-30 00:00:00 :: 2019-01-05 23:59:00":"2019-12-22 00:00:00 :: 2019-12-28 23:59:00"].copy()
l = [Weeks2017, Weeks2018, Weeks2019]
totalWeeks = pd.concat(l)

#-> De 05/11/2017 a 20/01/2018
#-> De 04/02/2018 a 27/10/2018
#-> De 04/11/2018 a 16/02/2019

firstTimeSeries = dataToResample.loc["2017-11-05 00:00:00":"2018-01-20 23:59:00"][["VB1", "VB2","VB3"]]
secondTimeSeries = dataToResample.loc["2018-02-04 00:00:00":"2018-10-27 23:59:00"][["VB1", "VB2","VB3"]]
thirdTimeSeries = dataToResample.loc["2018-11-04 00:00:00":"2019-02-16 23:59:00"][["VB1", "VB2","VB3"]]

#Estatísticas gerais das séries temporais definidas
firstTimeSeries.describe()
secondTimeSeries.describe()
thirdTimeSeries.describe()

#Caraterização em termos de valores omissos das séries temporais definidas
firstTimeSeries.isnull().sum()
secondTimeSeries.isnull().sum()
thirdTimeSeries.isnull().sum()

#secondTimeSeries.ffill(inplace = True) #Preenchimento dos valores omissos realizado tendo por base o último valor registado

secondTimeSeries.interpolate(method='linear', inplace=True)

def processStatisticsWeekly(data, key):
    #key -> Representa a variável de análise que se quer obter estatísticas
    indexWeekly = pd.date_range(start = "2018-02-04 00:00:00", end = "2018-10-28 00:00:00", freq = "W")
    size = len(indexWeekly) - 1
    minute = Minute(1)
    weeklyStatsValues = []

    for i in range(size):
        startWeek = indexWeekly[i]
        endWeek = indexWeekly[i + 1] - minute
        labelOfWeek = " :: ".join([str(startWeek), str(endWeek)])
        
        weekMedian = data.loc[startWeek:endWeek][key].mean()
        weekStd = data.loc[startWeek:endWeek][key].std()
        weekMax = data.loc[startWeek:endWeek][key].max()
        weekMin = data.loc[startWeek:endWeek][key].min()
        weekKurtosis = data.loc[startWeek:endWeek][key].kurt()
        weekCofVariation = weekStd / weekMedian
        weekAmpl = weekMax - weekMin
        weekStationarity = adfuller(data.loc[startWeek:endWeek][key])[1]
        
        values = {"Média": weekMedian, "Desvio-Padrão": weekStd, "Cof. Variação": weekCofVariation, "Máx": weekMax,
                  "Min": weekMin, "Amplitude": weekAmpl, "Kurtosis": weekKurtosis, "p-Value": weekStationarity}

        df = pd.DataFrame(data = values,
                          index = [labelOfWeek],
                          columns = ["Média", "Desvio-Padrão", "Cof. Variação", "Máx",
                                     "Min", "Amplitude", "Kurtosis", "p-Value"])
        
        weeklyStatsValues.append(df)

    dfWeekly = pd.concat(weeklyStatsValues)
    return dfWeekly

statsVB1 = processStatisticsWeekly(secondTimeSeries, "VB1")
statsVB2 = processStatisticsWeekly(secondTimeSeries, "VB2")
statsVB3 = processStatisticsWeekly(secondTimeSeries, "VB3")

# Processo de criação de um gráfico com os dados e respetivos labels dos mesmos utilizando o matplotlib
labels = ["2018-02-04", "2018-02-11", "2018-02-18", "2018-02-25", "2018-03-04", "2018-03-11", "2018-03-18", "2018-03-25",
          "2018-04-01", "2018-04-08", "2018-04-15", "2018-04-22", "2018-04-29", "2018-05-06", "2018-05-13", "2018-05-20",
          "2018-05-27", "2018-06-03", "2018-06-10", "2018-06-17", "2018-06-24", "2018-07-01", "2018-07-08", "2018-07-15",
          "2018-07-22", "2018-07-29", "2018-08-05", "2018-08-12", "2018-08-19", "2018-08-26", "2018-09-02", "2018-09-09",
          "2018-09-16", "2018-09-23", "2018-09-30", "2018-10-07", "2018-10-14", "2018-10-21"]

fig, ax = plt.subplots()
ax.plot(statsVB1.loc[:]["p-Value"], color = "blue", label = "VB1 (mm/s)")
ax.plot(statsVB2.loc[:]["p-Value"], color = "orange", label = "VB2 (mm/s)")
ax.plot(statsVB3.loc[:]["p-Value"], color = "green", label = "VB3 (mm/s)")
ax.legend()
ax.set(title = "Evolução do p-value", xlabel = "Semanas")
plt.xticks(np.arange(len(labels)), labels)
fig.autofmt_xdate()
plt.show()

#Criação de um gráfico de linhas
plt.figure(figsize = (12,6))
sns.lineplot(x = secondTimeSeries.index, y = "VB1", data = secondTimeSeries.loc[:, ["VB1"]])
plt.title("VB1 (mm/s)")
plt.show()
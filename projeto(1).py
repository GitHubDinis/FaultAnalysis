import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pandas.tseries.offsets import Minute
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from pandas.tseries.offsets import Day

#Processo de serialização e desserialização dos dados a serem analisados utilizando o pickle
filename = "C:/Users/Utilizador/Downloads/Material_Projeto/Material_Projeto/dataProcessed"
#pickle.dump(data2017, open(filename, "wb"))
data = pickle.load(open(filename, "rb"))

data.loc["2017"][["VB1", "VB2", "VB3"]].describe() #Segmentação anual das variáveis a serem analisadas
data.loc["2018"][["VB1", "VB2", "VB3"]].describe() #Segmentação anual das variáveis a serem analisadas
data.loc["2019"][["VB1", "VB2", "VB3"]].describe() #Segmentação anual das variáveis a serem analisadas

def processDataDaily(data):
    indexDaily = pd.date_range(start="2017-01-01 00:00:00", end="2019-12-31 23:59:00", freq="D")
    size = len(indexDaily) - 1
    minute = Minute(1)
    dailyNanValues = []

    for i in range(size):
        startDay = indexDaily[i]
        endDay = indexDaily[i + 1] - minute
        labelOfDay = " :: ".join([str(startDay), str(endDay)])
        
        vb1 = data.loc[startDay:endDay][["VB1"]].isnull().sum()["VB1"]
        vb2 = data.loc[startDay:endDay][["VB2"]].isnull().sum()["VB2"]
        vb3 = data.loc[startDay:endDay][["VB3"]].isnull().sum()["VB3"]
        values = {"VB1": vb1, "VB2": vb2, "VB3": vb3}

        df = pd.DataFrame(data=values,
                          index=[labelOfDay],
                          columns=["VB1", "VB2", "VB3"])
        
        dailyNanValues.append(df)

    dfDaily = pd.concat(dailyNanValues)
    return dfDaily

daysProcessed = processDataDaily(data)

dataToResample = data.copy()
dataToResample = dataToResample.resample("5min", closed = "right", label = "right").mean() #Downsampling

daysResampled = processDataDaily(dataToResample)

firstTimeSeries = dataToResample.loc["2017-11-05 00:00:00":"2018-01-20 23:59:00"][["VB1", "VB2","VB3"]]
secondTimeSeries = dataToResample.loc["2018-02-04 00:00:00":"2018-10-27 23:59:00"][["VB1", "VB2","VB3"]]
thirdTimeSeries = dataToResample.loc["2018-11-04 00:00:00":"2019-02-16 23:59:00"][["VB1", "VB2","VB3"]]

firstTimeSeries.describe()
secondTimeSeries.describe()
thirdTimeSeries.describe()

firstTimeSeries.isnull().sum()
secondTimeSeries.isnull().sum()
thirdTimeSeries.isnull().sum()

secondTimeSeries.interpolate(method='linear', inplace=True)

def processStatisticsDaily(data, key):
    indexDaily = pd.date_range(start="2018-02-04 00:00:00", end="2018-10-28 00:00:00", freq="D")
    size = len(indexDaily) - 1
    minute = pd.Timedelta(minutes=1)
    dailyStatsValues = []

    for i in range(size):
        startDay = indexDaily[i]
        endDay = indexDaily[i + 1] - minute
        labelOfDay = " :: ".join([str(startDay), str(endDay)])

        dayData = data.loc[startDay:endDay][key]

        if not dayData.var():
            # Se a variância da série for zero, não há necessidade de calcular a estatística de estacionaridade
            dayStationarity = np.nan
        else:
            dayStationarity = adfuller(dayData)[1]

        values = {"Média": dayData.mean(), "Desvio-Padrão": dayData.std(), "Cof. Variação": dayData.std() / dayData.mean(),
                  "Máx": dayData.max(), "Min": dayData.min(), "Amplitude": dayData.max() - dayData.min(),
                  "Kurtosis": dayData.kurt(), "p-Value": dayStationarity}

        df = pd.DataFrame(data=values,
                          index=[labelOfDay],
                          columns=["Média", "Desvio-Padrão", "Cof. Variação", "Máx",
                                   "Min", "Amplitude", "Kurtosis", "p-Value"])

        dailyStatsValues.append(df)

    dfDaily = pd.concat(dailyStatsValues)
    return dfDaily

statsVB1 = processStatisticsDaily(secondTimeSeries, "VB1")
statsVB2 = processStatisticsDaily(secondTimeSeries, "VB2")
statsVB3 = processStatisticsDaily(secondTimeSeries, "VB3")

# Criação dos Gráficos da Variação do P-Value
plt.figure(figsize=(10, 6))
plt.plot(statsVB3.loc['2018-02-28':'2018-04-01']["p-Value"], color = "green", label = "VB1 (mm/s)")
plt.title('P-Value Variation (2018-03)')
plt.xlabel('Date')
plt.xticks(ticks=range(1, 32), labels=range(1, 32))
plt.ylabel('p-value')
plt.grid(True)
plt.show()
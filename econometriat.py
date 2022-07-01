#!/usr/bin/env python
# coding: utf-8

# In[121]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# Lendo os arquivos

# In[5]:


cafe = pd.read_excel(r"C:\Users\João Vitor\Desktop\econometria\CAFE.xls")
milho = pd.read_excel(r"C:\Users\João Vitor\Desktop\econometria\MILHO.xls")
soja = pd.read_excel(r"C:\Users\João Vitor\Desktop\econometria\SOJA CE.xls")


# In[3]:


cafe


# In[6]:


milho


# In[7]:


soja


# fazendo uma função para pegar os valores da diferença do ln do preço

# In[4]:


def transformação(data):
    data = data.iloc[:,:-1]
    data.reset_index(drop = True , inplace=True)
    data.columns = ['Data' , 'preço']
    data['lnpreçod'] = np.log(data['preço']).diff()
    return data


# In[10]:


cafet = transformação(cafe)
milhot = transformação(milho)
sojat = transformação(soja)


# In[9]:


cafet


# In[11]:


milhot


# In[12]:


sojat


# In[269]:


from statsmodels.tsa.seasonal import seasonal_decompose


# Função para decomposição

# In[270]:


def decomposição(data):
    fig, axes = plt.subplots(4, 1, figsize = (13,10))
    decomp = seasonal_decompose(data['preço'], model="additive", freq = 24) 
    sns.lineplot(x=range(len(data.iloc[:,1])), y=data.iloc[:,1] , ax=axes.reshape(-1)[0])
    sns.lineplot(x=range(len(data.iloc[:,1])), y=decomp.trend , ax=axes.reshape(-1)[1])
    sns.lineplot(x=range(len(data.iloc[:,1])), y=decomp.seasonal , ax=axes.reshape(-1)[2])
    sns.lineplot(x=range(len(data.iloc[:,1])), y=decomp.resid , ax=axes.reshape(-1)[3])


# In[271]:


decomposição(cafet)


# In[272]:


decomposição(milhot)


# In[273]:


decomposição(sojat)


# Pela verificação gráfica, todas séries apresentam uma tendência no seu preço padrão, essa tendência deve ser retirada

# In[235]:


from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf


# Função de autocorrelação e autocorrelação parcial para saber a ordem máxima nos modelos arima

# In[236]:


def pacfacf (data):
    plot_acf(data['lnpreçod'][1:]);
    plot_pacf(data['lnpreçod'][1:]);


# In[237]:


pacfacf(cafet)


# Pelo gráfico , a ordem máxima de ar e ma para o retorno do café é 1,1

# In[238]:


pacfacf(milhot)


# Já para o milho se parece mais com um ARMA(5,10) no máximo

# In[239]:


pacfacf(sojat)


# (2,2) para a soja

# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")


# Função para plotar o preço e o retorno , respectivamente

# In[24]:


def timeplot(data):
    fig, axes = plt.subplots(2, 1, figsize = (10,10))
    sns.lineplot(x=range(len(data.iloc[:,1])), y=data.iloc[:,1] , ax=axes.reshape(-1)[0])
    sns.lineplot(x=range(len(data.iloc[1:,2])), y=data.iloc[1:,2] , ax=axes.reshape(-1)[1])


# In[25]:


timeplot(sojat)


# In[26]:


timeplot(cafet)


# In[27]:


timeplot(milhot)


# Pelos gráficos, todas parecem não estácionarias na forma de preço, mas estacionárias no retorno

# In[28]:


from statsmodels.tsa.stattools import adfuller
from arch.unitroot import PhillipsPerron


# teste Dickey Fuller e Phillip Perron, os primeiros valores correspondem ao valor estatístico

# In[217]:


adfuller(cafet['lnpreçod'][1:] , regression = 'nc' , maxlag = 0)


# In[39]:


PhillipsPerron(cafet['lnpreçod'][1:] , lags = 0)


# In[218]:


adfuller(milhot['lnpreçod'][1:] , regression = 'nc' , maxlag = 0)


# In[42]:


PhillipsPerron(milhot['lnpreçod'][1:] , lags = 0)


# In[219]:


adfuller(sojat['lnpreçod'][1:] , regression = 'nc' , maxlag = 0)


# In[44]:


PhillipsPerron(sojat['lnpreçod'][1:] , lags = 0)


# Todas estatísticas são menores que os valores críticos, ou seja, podemos rejeitar a hipótese nula de raiz unitária e estimar as séries ARMA

# In[46]:


from statsmodels.tsa.arima_model import ARIMA


# Função que retorna os valores aic e bic

# In[114]:


def aicbic(data , ar=0 , ma=0):
    aicbicframe = pd.DataFrame()
    processos = []
    aic = []
    bic = []
    for i in range(ar):
        for k in range(ma):
            try:
                model = ARIMA(data['lnpreçod'][1:],order = (i,0,k)).fit(trend='nc')
                processos.append('ARMA ' + str(i) + ' ' + str(k))
                aic.append(model.aic)
                bic.append(model.bic)
            except:
                processos.append('ARMA ' + str(i) + ' ' + str(k))
                aic.append(np.nan)
                bic.append(np.nan)
    aicbicframe['processos'] = processos
    aicbicframe['aic'] = aic
    aicbicframe['bic'] = bic
    return aicbicframe


# Função para estimar ARIMA

# In[128]:


def arima(data , ar , ma):
    model = ARIMA(data['lnpreçod'][1:],order = (ar,0,ma)).fit(trend='nc')
    return print(model.summary())


# In[115]:


cafeab = aicbic(cafet , 4 , 4)


# In[118]:


cafeab.sort_values('aic' , ascending=True)


# Como a tabela mostra, os valores de aic estão muito próximos, e pela fac e facp, os valores máximos são 1,1. Então, por simplicidade, a estimação será um ARMA(2,1)

# In[274]:


arima(cafet , 2,1)


# In[275]:


milhoab = aicbic(milhot , 5 , 10)


# In[276]:


milhoab.sort_values('aic' , ascending=True)


# In[277]:


arima(milhot , 2 , 3)


# In[124]:


sojaab = aicbic(sojat , 4 , 4)


# In[125]:


sojaab.sort_values('aic' , ascending=True)


# In[132]:


arima(sojat , 2, 1)


# # Agora, tentativa de reprodução do trabalho de Walberti Saith e Eder Luís Tomokazu Kamitani

# valores tomados de 03/01/2005 até 31/01/2011

# In[202]:


def dataj(data):
    return data.iloc[data[data['Data'] == '03/01/2005'].index.values[0]:data[data['Data'] == '31/01/2011'].index.values[0]+1]  


# In[209]:


cafej = dataj(cafet)
milhoj = dataj(milhot)
sojaj = dataj(sojat)


# In[206]:


cafej


# In[210]:


milhoj


# In[211]:


sojaj


# In[240]:


pacfacf(cafej)


# In[241]:


pacfacf(milhoj)


# In[242]:


pacfacf(sojaj)


# In[207]:


timeplot(cafej)


# In[212]:


timeplot(milhoj)


# In[213]:


timeplot(sojaj)


# In[216]:


adfuller(cafej['lnpreçod'][1:] , regression = 'nc' , maxlag = 0)


# In[215]:


PhillipsPerron(cafej['lnpreçod'][1:] , lags = 0)


# In[220]:


adfuller(milhoj['lnpreçod'][1:] , regression = 'nc' , maxlag = 0)


# In[221]:


PhillipsPerron(milhoj['lnpreçod'][1:] , lags = 0)


# In[223]:


adfuller(sojaj['lnpreçod'][1:] , regression = 'nc' , maxlag = 0)


# In[224]:


PhillipsPerron(sojaj['lnpreçod'][1:] , lags = 0)


# In[225]:


cafeabjj = aicbic(cafej , 4 , 4)


# In[226]:


cafeabjj.sort_values('aic' , ascending=True)


# In[227]:


arima(cafej , 2 , 2)


# In[228]:


milhoabjj = aicbic(milhoj , 4 , 4)


# In[230]:


milhoabjj.sort_values('aic' , ascending=True)


# In[231]:


arima(milhoj , 3 , 3)


# In[232]:


sojaabjj = aicbic(sojaj , 4 , 4)


# In[233]:


sojaabjj.sort_values('aic' , ascending=True)


# In[234]:


arima(sojaj , 1 , 0)


# In[ ]:





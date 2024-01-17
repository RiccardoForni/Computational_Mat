
import function as rf
import plotting as rp
import pandas as pd
import numpy as np
import random



def give_stock(stock,q):
    temp_dict = {name: 0 for name in stock}
    # Lista per tracciare i numeri casuali già estratti
    numeri_già_estratti = []

    for _ in range(q):
        # Generazione di un valore random intero da 0 a 10
        indice_random = random.randint(0, 10)

        # Riesecuzione se il numero è già stato estratto
        while indice_random in numeri_già_estratti:
            indice_random = random.randint(0, 10)

        # Aggiunta del numero all'elenco di quelli estratti
        numeri_già_estratti.append(indice_random)

        # Cambio del valore corrispondente all'indice random da 0 a 1 nel dizionario
        chiavi = list(temp_dict.keys())
        if 0 <= indice_random < len(chiavi):
            chiave_casuale = chiavi[indice_random]
            temp_dict[chiave_casuale] = 1

    return temp_dict


def create_ILP(df,df_corr,y,q,stock_esg,esg):
    C = np.array([]) 
    A = np.array([])
    b = np.array([])
    #define C
    for i in df_corr.index:
        temp = np.array([])
        for j  in df_corr.columns:
            temp = np.append(temp,df_corr.loc[i,j])
            
        C = np.append(C,temp,axis=0)
    #define A,b
    # m_factor = np.array([1,1,1])
    
    m_factor = np.array([])
    m_factor = np.append(m_factor,np.ones(df_corr.shape[1],dtype=int))
    m_factor = np.append(m_factor,np.zeros(df_corr.shape[0]*df_corr.shape[1]-df_corr.shape[1],dtype=int))
    
    for i in range(df_corr.shape[0]):
        b = np.append(b,1)
        for z in m_factor:
            A = np.append(A,z*1)

        m_factor= np.roll(m_factor, df_corr.shape[1])
    for key,valore in y.items():
        A = np.append(A,valore)  
      
    b = np.append(b,q)
    for i in df_corr.index:
        for j in df_corr.columns:
            b = np.append(b,y[j])
            if y[j] == 0:
                A = np.append(A,0)
            else:
                for i in list(y.values()):
                    A = np.append(A,y[j]*i)
    

    for j in stock_esg.columns:
        print(stock_esg[j]*y[j])
        A = np.append(A,stock_esg[j]*y[j])

    b = np.append(b,esg*q)



    return 0 

try:
    Symbol = ["Date","IBM","NKE","GE","GS","SBUX","JNJ","AVGO","LRCX","MMC","ROST"]
    time = "2013/9/30"
    df_stocks,n_stock, Symbol = rf.take_data(Symbol,time)
    df_stocks.to_excel("Stocks.xlsx")
except:
    print("cannot download, take directly the file is possible")
    df_stocks = pd.read_excel("Stocks.xlsx")
    Symbol = df_stocks.columns
    n_stock = len(Symbol)

check = True
while check:
    q = input("Select number of stocks")
    if int(q) <= n_stock and int(q)> 0:
        check = False

q = int(q)

esg_target = input("Esg target values -> 0 to 40+")
esg_target =  int(esg_target)
    
del check        

df_esg = pd.read_excel("ESG.xlsx")

#data
dates = df_stocks.iloc[:,0]
values = df_stocks.iloc[:,1:]


#values = values.iloc[:,list(range(0,q))]

#Correlation 
corr_matrix = values.corr()
rp.plot_corr(corr_matrix)

stock = give_stock(values.columns,q)

create_ILP(values,corr_matrix,stock,q,df_esg,esg_target)



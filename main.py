
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

def scale(df):
    m_factor = np.array([])
    m_factor = np.append(m_factor,np.ones(df.shape[1],dtype=int))
    m_factor = np.append(m_factor,np.zeros(df.shape[0]*df.shape[1]-df.shape[1],dtype=int))
    return m_factor

def create_ILP(df,df_corr,y,q,stock_esg,esg):
    C = np.array([]) #function to optimize
    A = []
    b = np.array([])
    #define C
    for i in df_corr.index:
        temp = np.array([])
        for j  in df_corr.columns:
            temp = np.append(temp,df_corr.loc[i,j])
            
        C = np.append(C,temp,axis=0)

    #C = np.array(df_corr)
    #define A,b
    m_factor = scale(df_corr)

    for i in range(df_corr.shape[0]):
        t = np.array([])
        b = np.append(b,1)
        for z in m_factor:
            t = np.append(t,z*1)
        A.append(t)

        m_factor= np.roll(m_factor, df_corr.shape[1])

    t = np.array([])
    for i in df_corr.index:
        for j in df_corr.columns:
            if df_corr.loc[i,j] == 1:
                    if y[i] == y[j]:
                        t = np.append(t,y[i]*y[j])
            else:
                t=np.append(t,0)
                    

    A.append(t)
    b = np.append(b,q)

    m_factor = scale(df_corr)
    #3rd constrait
    for j in df_corr.columns:
        t = np.array([])
        b = np.append(b,y[j]*df_corr.shape[1])
        for z in m_factor:
            t = np.append(t,z*1)
        A.append(t)

        m_factor= np.roll(m_factor, df_corr.shape[1])

    
    #last
    t = np.array([])
    for i in df_corr.index:
        for j in df_corr.columns:
            if df_corr.loc[i,j] == 1:
                    if y[i] == y[j]:
                        t = np.append(t,stock_esg[j]*y[j])
            else:
                t=np.append(t,0)
    A.append(t)

    b = np.append(b,esg*q)
    A = np.array(A)
    A = A*-1
    b = b*-1
    C = C*-1

    pd.DataFrame(A).to_excel("A.xlsx")
    pd.DataFrame(b.T).to_excel("b.xlsx")
    pd.DataFrame(C.T).to_excel("C.xlsx")

    return A,b,C

try:
    print("take data from file")
    df_stocks = pd.read_excel("Stocks.xlsx")
    Symbol = df_stocks.columns
    n_stock = len(Symbol)

 
except:
    try:
        print("cannot find file, download data")
        Symbol = ["Date","IBM","NKE","GE","GS","SBUX","JNJ","AVGO","LRCX","MMC","ROST"]
        time = "2013/9/30"
        df_stocks,n_stock, Symbol = rf.take_data(Symbol,time)
        df_stocks.to_excel("Stocks.xlsx")
        df_stocks = df_stocks.drop("Date")
    except:
        print("not possible to continue")


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



import matplotlib.pyplot as plt
import seaborn as sns
"""
["BWA", "QQQ", "PAY", "HSY", "HWM", 
"LKQ", "HOL", "LVS", "MCK", "KMX", 
"QRV", "VIC", "AEE", "FIT", "BEN",
"MRN", "PAN", "SRE", "TRV", "WST",
"AIG", "CCI", "CPR", "MTC", "AVY",
"BXP", "BBW", "GPC", "PEG", "PFE",
"WMT", "CPT", "GEN", "DUK", "KMB", 
"LUL", "CAR", "AJG", "IPG", "RSG",
"CHT", "XYL", "APA", "VLO", "UHS",
"MTD", "ADP", "SHW", "AME", "LHX",
"WRB", "EXP", "MHK", "CSG", "COR",
"TER", "BAC", "AMA", "ITW", "AFL", 
"CNC", "EOG", "TGT", "ORC", "HIG", 
"TRG", "JPM", "VRS", "DVA", "GWW", 
"MRO", "WTW", "MTB", "JKH", "MAR", 
"PSA", "OKE", "DPZ", "EQR", "CNP", 
"WFC", "ELV", "KVU", "PXD", "FLT", 
"STL", "IDX", "DOW", "APD", "DOV", 
"DRI", "BLK", "FAN", "TYL", "CTS", 
"REG", "ANS", "PNC", "KMI", "CEG"]
["Date",
"IBM","NKE","GE","GS","SBUX",
"JNJ","AVGO","LRCX","MMC","ROST",
"CMCSA","WFC","VZ","AMGN","QCOM",
"SPGI","UBER","NEE","PLD","SYK",
"FOXA","FRT","GNRC","FMC","CMA",
"ZION","DVA","RL","VFC","NWS",
"AIZ","CTLT","WYNN","PAYC","HSIC",
"APA","UHS","QRVO","AOS","MTCH"]
"""
def plot_corr(matrix):
    sns.heatmap(matrix, cmap="Greens",annot=True)
    plt.savefig("correlation matrix.png")
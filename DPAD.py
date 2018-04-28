
#import quandl
#import random
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
import time
from tkinter import*
from tkinter import ttk
import tkinter as tk
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
import webbrowser
style.use('ggplot')

def quit_a():
    root.destroy()

def download():
    new = 2
    url= "https://www.google.com/finance"
    webbrowser.open(url,new)
    
def main_file():
    df = pd.read_csv(root.fileName, index_col='Date', parse_dates=True)
    
   
    
    
    df = df[['Open',  'High',  'Low',  'Close', 'Volume']]
    df['HL_PCT'] = (df['High'] - df['Low']) / df['Low'] * 100.0
    df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
    
    df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume']]
    forecast_col = 'Close'
  
    forecast_out = int(math.ceil(0.01 * len(df)))
    df['label'] = df[forecast_col].shift(-forecast_out)
    #df['label'] = df[forecast_col].shift(periods=forecast_out)
      ##drop any NAN from data frame, define X as feature and Y as label
    
    X = np.array(df.drop(['label'], 1))
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]
    
    df.dropna(inplace=True)
    
   ## y = np.array(df['label'])
    y=df['label']
    y.dropna(inplace=True)
    y=np.array(y)    
    
    #Training
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    #confidence = random.uniform(80,98)
    
    ##print(df.head())
    print("Accuracy=",confidence*100,"%")
    
    forecast_set = clf.predict(X_lately)
    df['Forecast'] = np.nan
    
    last_date = df.iloc[-1].name
    ##last_date =  df.iloc[0].nam
    #last_unix = time.mktime(last_date.timetuple())
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day
    
    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
    
    df['Close'].plot()
    df['Forecast'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()   
   # plt.plot(forecast_set)
    #plt.show()    

    
def use_exist():
    root.fileName = askopenfilename( filetypes = ( ("CSV Files", "*.csv"), ("All files","*.*"))) 
    main_file()
    
root= Tk()
root.title=("DPAD")
frameback = Frame(root, bg='black', height=1000, width=1000)
frameback.pack()

frametop = Frame(frameback, bg='red')
frametop.pack(side=TOP, fill='both')

photo = PhotoImage(file="logo.png")
map=Label(frametop, image=photo)
map.photo=photo
map.pack()


frame= Frame(frametop, bg='black')
frame.pack(side=BOTTOM, fill="both")

importbutton = Button(frame, text="Import Existing Data", bg="grey", fg="black")
importbutton.pack(side=LEFT, padx=55, pady=55)
importbutton.config(height=3, width=20, command = use_exist)
'''
manualbutton = Button(frame, text="MANUAL ENTRY", bg="grey", fg="black")
manualbutton.config(height=3, width=15)
manualbutton.pack(side=LEFT, padx=55, pady=55)
'''
manualbutton = Button(frame, text="Download stock data", bg="grey", fg="black")
manualbutton.pack(side=LEFT, padx=55, pady=55)
manualbutton.config(height=3, width=20, command = download)

quitbutton = Button(frame, text="QUIT", bg="grey", fg="black", command=quit)
quitbutton.config(height=3, width=15, command = quit_a)
quitbutton.pack(side=LEFT, padx=55, pady=55)

root.mainloop()
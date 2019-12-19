import pandas as pd 
import csv
import pymongo
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
cors = CORS(app)

import pandas as pd
import numpy as np
import math
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.utils import to_categorical

from imblearn.datasets import make_imbalance
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced
from sklearn.svm import SVR 
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import  mutual_info_regression
import math

indices = []

client = pymongo.MongoClient("mongodb+srv://jgirlsdad:444jayla@cluster0-dgjk9.mongodb.net/test?retryWrites=true&w=majority")
mydb = client.climate
mycol = mydb["monthly"]

print ("finished connecting to MDB")

mydoc = mycol.find({"city":"DENVER"})
hist = {}
variable = "TMAX"
for x in mydoc:  
    temp = []
    yrmos = []
    hist = {}
    for key in x[variable]:
        spl = key.split("-")
        yr = spl[0]
        if yr in hist:
            hist[yr]+=1
        else:
            hist[yr]=1

    for yr in hist:
        print ("HIST ",yr,hist[yr])




temp = []
features = pd.DataFrame()
for year in range(1950,2019):
    for month in range(1,13):
        yrmo = year*100 + month
        temp.append(yrmo)
        
features['yearmo'] = temp


with open("data/file_list","r") as F:
    for file in F:
        file = file.rstrip("\n")
        spl = file.split(".")
        name = spl[0]
        temp = []
        with open(f"data/{file}","r") as fin:
            line=fin.readline()
            spl = line.split()
            start = int(spl[0])
            end = int(spl[1])
            for nn in range(start,end):
               line = fin.readline()
               spl = line.split()
               year = int(spl[0])
               if (year >= 1950 and year <= 2018):
                  for val in spl[1:]:
                    if float(val) != -99.99:
                      temp.append(val)
            if len(temp) == 828:
                features[name] = temp
                features[name] = features[name].astype('float')
print (features.head())  


def gridSearch(X,y):


    svr = GridSearchCV(
        SVR(kernel='rbf', gamma=0.1), cv=5,
        param_grid={"C": [1e0, 1e1, 1e2, 1e3],
        "gamma": np.logspace(-2, 2, 5)})

    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)

    X_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)

    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    y_train_scaled = y_scaler.transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    svr.fit(X_train_scaled, y_train_scaled)
    r2 = svr.score(X_test_scaled,y_test_scaled)
    train_score = svr.score(X_train_scaled,y_train_scaled)
    test_score = svr.score(X_test_scaled,y_test_scaled)
    temp ={}
    scores = []   
    temp["r2"] = r2
    temp["train"] = train_score
    temp["test"] = test_score
    scores.append(temp)

    return(scores)




def train(XX,yy,nsplit,toScale):
    ninst = len(yy)
    y=yy[0:ninst-nsplit]
    X=XX[0:ninst-nsplit]
    X_exp = XX[ninst-nsplit-1:]
    y_exp = yy[ninst-nsplit-1:]

    feature_info = ""
  #  feature_info = mutual_info_regression(X, y)
    

    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)
    
    X_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    if (toScale == 0):  # do not scale data
       return (X_train,X_test,X_exp,y_train,y_test,y_exp,feature_info,X_scaler,y_scaler)

    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    X_exp_scaled = X_scaler.transform(X_exp)

    y_train_scaled = y_scaler.transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)
    y_exp_scaled = y_scaler.transform(y_exp)

    
    return (X_train_scaled,X_test_scaled,X_exp_scaled,y_train_scaled,y_test_scaled,y_exp_scaled,feature_info,X_scaler,y_scaler)

def svrRegression(X,y,nphases):

    model = SVR(C=1.0,cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='scale',
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

    temp,feature_info = phaseShiftNew(model,"SVR",X,y,nphases)
   
    return temp,feature_info


def bayesianRegression(X,y,nphases):

    model = BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True,
       fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, n_iter=300,
       normalize=False, tol=0.001, verbose=False)

    temp,feature_info = phaseShiftNew(model,"Bayesian",X,y,nphases)

 #   r2,train_score,test_score = regression(model,"bayesian",X,y)

    return  temp,feature_info

def linearRegression(X,y,nphases):
    model = LinearRegression()

    temp,feature_info = phaseShiftNew(model,"Linear",X,y,nphases)
   
    return temp,feature_info


def regression(model,model_name,X,y):
    X_train_scaled,X_test_scaled,X_exp_scaled,y_train_scaled,y_test_scaled,y_exp_scaled,feature_info,X_scaler,y_scaler = train(X,y,24,1)

    model.fit(X_train_scaled, y_train_scaled)
    
    train_score = model.score(X_train_scaled,y_train_scaled)
    test_score = model.score(X_test_scaled,y_test_scaled)
    y_predict = model.predict(X_exp_scaled)
    r2 = r2_score(y_exp_scaled,y_predict)

    y_predict = y_scaler.inverse_transform(y_predict)
    y_exp_scaled = y_scaler.inverse_transform(y_exp_scaled)
    X_exp_scaled = X_scaler.inverse_transform(X_exp_scaled)
    bb = []
    for a in y_exp_scaled:
      bb.append(a[0])

    return r2,train_score,test_score,feature_info,bb,y_predict,X_exp_scaled

def phaseShiftNew(model,model_name,X,y,nphases):
    tempx = X
    tempy = y
    hold = []
    for nn in range(0,nphases+1):
   #     tempy = np.delete(tempy,0,0)
   #     tempx = np.delete(tempx,-1,0)
        tempy = y[nn:]
        if (nn > 0): 
          tempx = X[:0-nn]
        else:
          tempx = X
        print ("SHAPES ",tempy.shape,tempx.shape)

        (r2,train_score,test_score,feature_info,y_exp,y_pred,X_exp) = regression(model,model_name,tempx,tempy)
        temp ={}
        temp[nn] = {}
        mm=0
  #      print ("Y EXP",y_exp[0])
  #      print ("Y PRED ",y_pred)
       
        temp[nn]['feature_score'] = {}
        for feat in indices:
     #      print ("featurei nfo ",feat,feature_info[mm])
           temp[nn]['feature_score'][feat] = feature_info[mm]
           mm+=1
           
         
        temp[nn]["model"] = model_name
        temp[nn]["r2"] = r2
        temp[nn]["train"] = train_score
        temp[nn]["test"] = test_score
        
        temp[nn]['y_obs'] = y_exp
    #   temp[nn]['y_pred'] = []
        
        
        temp[nn]['y_pred'] = y_pred.tolist()
#        temp[nn]['x_obs'] = X_exp
        hold.append(temp)

    return (hold,feature_info)


def phaseShift(X,y,nphases):
    tempx = X
    tempy = y
    scores = []
    for nn in range(0,nphases+1):
        tempy = np.delete(tempy,0,0)
        tempx = np.delete(tempx,-1,0)
        (r2,train_score,test_score) = linearReg(tempx,tempy)
        temp ={}
        temp[nn] = {}
        temp[nn]["r2"] = r2
        temp[nn]["train"] = train_score
        temp[nn]["test"] = test_score
        scores.append(temp)

    return (scores)
        


def getData(years,flag):
    if (len(years) > 1):
        start = years[0]
        end   = years[1]
        if (flag == 0):  # get date range
          mydoc = mycol.find({'year': { '$gte': start, '$lte': end }},{'year':1,'name':1,'maxStrength':1,'month':1,'minPressure':1,'maxWind':1,})
        else:  # get all years >= first year in array
          mydoc = mycol.find({'year': { '$gte': start}},{'year':1,'name':1,'maxStrength':1,'month':1,'minPressure':1,'maxWind':1,})

    else:
        start = years[0]
        myquery = {"year":start}
        mydoc = mycol.find(myquery)
    #mydoc = mycol.find()
    storms = []
    for x in mydoc:
        storm = {}
        storm['name'] = x['name'].strip()
        storm['year'] = x['year']
        storm['maxWind'] = x['maxWind']
        storm['minPressure'] = x['minPressure']
        storm['maxStrength'] = x['maxStrength']
        storm['month'] = x['month']
        tracks = []
        if 'tracks' in x:
            for date,info in x['tracks'].items():            
                obs = []
                obs.append(info['lat'])
                obs.append(info['lon'])
                obs.append(info['type'])
                obs.append(info['wind'])
                obs.append(info['pressure'])
                obs.append(info['strength'])
                obs.append(date)
                tracks.append(obs)
            storm['track'] = tracks
        storms.append(storm)
    return storms

def phaseShiftClassify(model,model_name,X,y,nphases):
    
    hold = []
    for nn in range(0,nphases+1):
        tempy = y[nn:]
        if (nn > 0): 
          tempx = X[:0-nn]
        else:
          tempx = X
        
 #       print (model_name,nn,tempx[-20:].tolist())
        (r2,train_score,test_score,y_pred,y_exp) = classifier(model,model_name,tempx,tempy)
   #     print (model_name,nn,r2,train_score,test_score)
        temp ={}
        temp[nn] = {}
     #   mm=0
     #   temp[nn]['feature_score'] = {}
     #   for feat in indices:
     #      print ("featurei nfo ",feat,feature_info[mm])
     #      temp[nn]['feature_score'][feat] = feature_info[mm]
     #      mm+=1
           
         
        temp[nn]["model"] = model_name
        temp[nn]["r2"] = r2
        temp[nn]["train"] = train_score
        temp[nn]["test"] = test_score
        
        temp[nn]['y_obs'] = y_exp
    #   temp[nn]['y_pred'] = []
        
        temp[nn]['y_pred'] = y_pred
#        temp[nn]['x_obs'] = X_exp
#        print ("TEMP ",temp)
        hold.append(temp)

    return (hold)

def regLogistic(X,y,nphases):

#    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)
    model = LogisticRegression()
    temp = phaseShiftClassify(model,"Logistic",X,y,nphases)

    return temp



def randomForestClassification(X,y,nphases):

   
    model = RandomForestClassifier(n_estimators=200)
    temp = phaseShiftClassify(model,"RFC",X,y,nphases)

    return temp




def svcClassification(X,y,nphases):

    model = SVC(kernel='linear')

    temp = phaseShiftClassify(model,"SVC",X,y,nphases)

    return temp


def classifier(model,model_name,X,y):
    
    X_train_scaled,X_test_scaled,X_exp_scaled,y_train_scaled,y_test_scaled,y_exp_scaled,feature_info,X_scaler,y_scaler = train(X,y,24,0)
    
    model.fit(X_train_scaled, y_train_scaled)

    train_score = model.score(X_train_scaled, y_train_scaled)
    test_score = model.score(X_test_scaled, y_test_scaled)
   
    y_predict = model.predict(X_exp_scaled)
    print (model_name," YPRED ",X_exp_scaled)


    bb = []
    for a in y_predict:
      bb.append(int(a))
    y_predict = bb


    cc = []
    for a in y_exp_scaled:
      cc.append(int(a[0]))

    y_exp_scaled = cc
 #   print ("YOBS ",y_exp_scaled)
 #   y_predict = y_scaler.inverse_transform(y_predict)
 #   y_exp_scaled = y_scaler.inverse_transform(y_exp_scaled)
 #   X_exp_scaled = X_scaler.inverse_transform(X_exp_scaled)

    good=0
    bad=0

    for mm in range(0,len(y_predict)):
   #     print (model_name,mm,y_predict[mm],y_exp_scaled[mm])
        if (y_predict[mm] == y_exp_scaled[mm]):
            good+=1
        else:
            bad+=1
           
   
    pp =  good / (good+bad)
#    print ("Good: ",good,"  Bad: ",bad,"  PP: ", pp)

   

    return pp,train_score,test_score,y_predict,y_exp_scaled



@app.route("/classify/<city>/<variable>/<ind>/<classType>/<numBins>") #/getdata/___/___
def classify(city,variable,ind,classType,numBins):
    if (classType == "PBIN"):
       ctype = 1
    elif (classType == "CBIN"):
       ctype = 2


  
    indices = ind.split(",")
    nphases = 12
#    for val in temp:
#        indices.append(val)
    
    mydoc = mycol.find({"city":city})
    data = pd.DataFrame()
    features['yearmo'] = features['yearmo'].astype(int)
  #  print (features.info())


    for x in mydoc:  
        temp = []
        yrmos = []
        ts = {}
        stats = {}
        for nn in range(1,13):
            stats[nn] = {}
        for key in x[variable]:
            value = float(x[variable][key])
            yrmo = key.replace("-","")
            mo = int(yrmo[4:6])
            if "min" in stats[mo]:
                if value < stats[mo]['min']:
                    stats[mo]['min'] = value
                if value > stats[mo]['max']:
                    stats[mo]['max'] = value
            else:
                stats[mo]['max'] = value
                stats[mo]['min'] = value

            yrmos.append(int(yrmo))
            temp.append(x[variable][key])
            ts[yrmo] = x[variable][key]

        start = min(yrmos)
        end = max(yrmos)    
        
        s1 = variable + " MEANS" 
        s2 = variable + " STDEV"   
        data[variable] = temp
        start = min(yrmos)
        end = max(yrmos)
        
    temp = pd.DataFrame()
    for val in indices:
      val = val.lower()
      temp[val] = features[val][(features["yearmo"] >= start) & (features['yearmo'] <= end)].astype(float)
    
    nl = len(indices)
    X = temp.values.reshape(-1,nl) 
   



    for mm in stats:
        stats[mm]['range'] = stats[mm]['max'] - stats[mm]['min']
                

    if ctype == 1:   #  use pre-defined bins bases on stdev
            yrmos = []  
            bins = []
            for key,value in ts.items():
                year = key[0:4]
                mo   = int(key[4:6])
                yrmos.append(key)

                mean = float(x[s1][mo-1])
                sdev = float(x[s2][mo-1])
                value = float(value)
                if value < (mean - 3*sdev):
                    bin = 1
                elif value < (mean-2*sdev):
                    bin=2
                elif value < (mean-sdev):
                    bin=3
                elif value < (mean+sdev):
                    bin=4
                elif value < (mean+2*sdev):
                    bin = 5
                elif value < (mean + 3*sdev):
                    bin = 6
                elif value >= (mean + 3*sdev):
                    bin = 7
                bins.append(int(bin))
    elif ctype == 2:  # user defined bins
        
            nbins = int(numBins)
            hist = {}
            yrmos = []
            bins = []
            for key,value in ts.items():
                year = key[0:4]
                mo   = int(key[4:6])
                go=1
                nn=0
                yrmos.append(key)
                scale = (stats[mo]['max'] - stats[mo]['min'])/nbins
                bin=-1
                value = float(value)
                while go and nn <= nbins:
                    nn+=1
                    if value < (stats[mo]['min']+ nn*scale) :
                        bin=nn
                        go=0
                bins.append(int(bin))
                if bin in hist:
                    hist[bin]+=1
                else:
                    hist[bin]=1
       #         print ("BIN ",mo,value,bin,stats[mo]['min'],stats[mo]['max'],scale)
            

    temp = pd.DataFrame(bins)
    y = temp.values.reshape(-1,1)
    
    output = []

    temp = regLogistic(X,y,nphases )
    output.append(temp)  

    temp = svcClassification(X,y,nphases )
    output.append(temp)  

    temp = randomForestClassification(X,y,nphases)
    output.append(temp)


    hld = {}
    hld['yearmo'] = yrmos
   
#   output.append(hld)
     
    return jsonify(output)


@app.route("/machine/<city>/<variable>/<machine>/<ind>") #/getdata/___/___
def machine(city,variable,machine,ind):
    
    indices = ind.split(",")
#    for val in temp:
#        indices.append(val)
    print ("machine learn ",city,variable,machine,indices)
    mydoc = mycol.find({"city":city})
    data = pd.DataFrame()
    features['yearmo'] = features['yearmo'].astype(int)
  #  print (features.info())

    for x in mydoc:  
        temp = []
        yrmos = []
        for key in x[variable]:
            yrmo = key.replace("-","")
            yrmos.append(int(yrmo))
            temp.append(x[variable][key])
        data[variable] = temp
        start = min(yrmos)
        end = max(yrmos)
    
 #   print(data)
    y = data[variable].values.reshape(-1,1)
    temp = pd.DataFrame()
    for val in indices:
      val = val.lower()
      temp[val] = features[val][(features["yearmo"] >= start) & (features['yearmo'] <= end)].astype(float)
    
    nl = len(indices)
    X = temp.values.reshape(-1,nl) 
  #   print (X)
    print ("len ",X.shape,y.shape)
    scores = []
    temp,feature_info = svrRegression(X,y,12)
    scores.append(temp)

    temp,feature_info = linearRegression(X,y,12)
    scores.append(temp)

    temp,feature_info = bayesianRegression(X,y,12)
    scores.append(temp)
    print ("FI ",feature_info)
 #   print (scores)

 #   r2 = neuralNet(X,y)
 #   print ("Neural Net output ",r2)
    return jsonify(scores)



@app.route("/neural/<city>/<variable>/<ind>/<phase>/<layers>/<nodes>") #/getdata/___/___
def neural(city,variable,ind,phase,layers,nodes):
    phase = int(phase) 
    layers = int(layers)
    nodes = int(nodes)
    indices = ind.split(",")

    print ("machine learn ",city,variable,indices,phase,layers,nodes)
    mydoc = mycol.find({"city":city})
    data = pd.DataFrame()
    features['yearmo'] = features['yearmo'].astype(int)
  #  print (features.info())

    for x in mydoc:  
        temp = []
        yrmos = []
        for key in x[variable]:
            yrmo = key.replace("-","")
            yrmos.append(int(yrmo))
            temp.append(x[variable][key])
        data[variable] = temp
        start = min(yrmos)
        end = max(yrmos)
    
    print("DDDATA ",data)
    y = data[variable].values.reshape(-1,1).astype(float)
    temp = pd.DataFrame()
    for val in indices:
      val = val.lower()
      temp[val] = features[val][(features["yearmo"] >= start) & (features['yearmo'] <= end)].astype(float)
    
    nl = len(indices)
    X = temp.values.reshape(-1,nl).astype(float) 
#    print ("XXXXX  ",X)
 #   print ("len ",X.shape,y.shape)
    scores = []
     
    tempy = y[phase:]
 #   print ("TEMP Y",tempy)
    if (phase > 0): 
        tempx = X[:0-phase]
    else:
        tempx = X
  #  print ("TEMPX ",tempx)
    scores = neuralNet(tempx,tempy,layers,nodes)
 #   print ("SCORES ",scores)

    return jsonify(scores)




def neuralNet(X,y,layers,nodes):
    X_train_scaled,X_test_scaled,X_exp_scaled,y_train_scaled,y_test_scaled,y_exp_scaled,feature_info,X_scaler,y_scaler = train(X,y,24,1)

 #   X_train_scaled,X_test_scaled,X_exp_scaled,y_train_scaled,y_test_scaled,y_exp_scaled,feature_info = train(X,y,24)
    nd = X.shape
    ndims = nd[1]

    print ("Layers : ",layers,nodes)



    model = Sequential()
    for nn in range(1,layers):
      print ("Adding Layer ",nn)
      model.add(Dense(units=nodes, activation='relu', input_dim=ndims))

    model.add(Dense(units=1))
    
    model.compile(optimizer='adam',
                loss='mse',
                metrics=['mse'])

   

    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        epochs=100,
        shuffle=True,
        verbose=0,
        validation_split = 0.2
    )
    hist = pd.DataFrame(history.history)
#    print ("History ",hist.head())
    model.summary()
    y_predict = model.predict(X_exp_scaled)
    
    r2 = model.evaluate(X_exp_scaled,y_exp_scaled)

    print ("R2  R2 ",r2)
    mde=0
    y_predict = y_scaler.inverse_transform(y_predict)
    y_exp_scaled = y_scaler.inverse_transform(y_exp_scaled)
    for nn in range(0,len(y_exp_scaled)):
        mde+= (y_exp_scaled[nn][0]-y_predict[nn][0])**2
        print ("YYY ",y_exp_scaled[nn][0],y_predict[nn][0])
    mde=math.sqrt(mde/len(y_exp_scaled))
    
    print ("MDE MDE ",mde)

    bb = []
    hold = []
    for a in y_exp_scaled:
      bb.append(float(a[0]))

    cc = []
    for a in y_predict:
      cc.append(float(a[0]))
    
    temp = {}
    temp["r2"] = mde
    
    temp['y_obs'] = bb
    temp['y_pred'] = cc
     

 #   temp['acc'] = hist['acc'].values.tolist()
 #   temp['acc_val'] = hist['val_acc'].values.tolist()


    temp['mse'] = hist['mean_squared_error'].values.tolist()
    temp['mse_val'] = hist['val_mean_squared_error'].values.tolist()

    hold.append(temp)



  #  print ("R2 = ",r2)
    return hold


@app.route("/getdata/<name>") #/getdata/___/___
def getData(name):
    print ("getting data info for ",name)
    mydoc = mycol.find({"city":name})
    data = []
    for x in mydoc:  
        dat = {}
        for y in x:
            if y != "_id":
                dat[y] = x[y]
 #       dat['name'] = x['city'].strip()
 #       dat['lat'] = x['lat']
 #       dat['lon'] = x['lon']
 #       dat['start'] = x['start']
 #       dat['end'] = x['end']
    
        
        data.append(dat)
        print ("DATA TS",data)
    return jsonify(data)


@app.route("/indices/<name>") #/getdata/___/___
def getIndices(name):

    data = []
    dates = {}
    ts = {}
    
    dates['yearmo'] = list(features['yearmo'])
    ts['ts'] = list(features[name])
    data.append(dates)
    data.append(ts)
    

    return jsonify(data)




@app.route("/getcities")
def getCities():
    print ("getting city info")
    mydoc = mycol.find({},{'city': 1,'lat':1,'lon':1,'start':1,'end':1})
    cities = []
    for x in mydoc:
        
        city = {}
        city['name'] = x['city'].strip()
        city['lat'] = x['lat']
        city['lon'] = x['lon']
        city['start'] = x['start']
        city['end'] = x['end']
        cities.append(city)
    return jsonify(cities)

@app.route("/info")
def getInfo():
    statsarr = []
    for year in stats:
        stats[year]['year'] = year
        statsarr.append(stats[year])
 
    return jsonify(statsarr)



@app.route("/")
def home():
   
      return render_template("index.html")



@app.route('/result',methods = ['POST', 'GET'])
def result():
    hit=0
    if request.method == 'POST':
 #      result = request.form
       query_string = ""
       
#getInfo()

#getStorm("Irma")

if __name__ == "__main__":  
    app.run(debug=True)
#   BACK END       BACK END     BACK END

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

indices = []  #  contains climate indices

#  connect to MongoDB database to get climate City Climate Data
client = pymongo.MongoClient("mongodb+srv://jgirlsdad:@cluster0-dgjk9.mongodb.net/test?retryWrites=true&w=majority")
mydb = client.climate
mycol = mydb["monthly"]


#  This is just a check to be sure our DB connection works on start up
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




temp = []  # temporary list/dict used throught the program 
features = pd.DataFrame()  # this is a Pandas Dataframe that will hold the Climate Indices

#  add year-month pairs to the features dataframe that will match the available data
for year in range(1950,2019):
    for month in range(1,13):
        yrmo = year*100 + month
        temp.append(yrmo)
        
features['yearmo'] = temp

#  the Climate Indices are stored in flat files, so read them all in and store them in features
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
    ''' Perform a gridSearh Machine Learning Task... not used right now '''

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
    ''' create a Train, Test and Experiment datasets out of the input data. If toScale=1, then
    the data will also be scaled.  If toScale=0, the data will not be scaled.  The Experiment 
    time series will be the last nsplit values in the XX and yy lists. The Experiment time series 
    will be used for creating prediction scores'''

    ninst = len(yy)  # length of input time series
    y=yy[0:ninst-nsplit]  #  this will be used to create the Train and Test time sries for the dependant variables
    X=XX[0:ninst-nsplit]  #  this will be used to create the Train and Test time series for the independant variables
    X_exp = XX[ninst-nsplit-1:]  # create the X experimental dataset for predictions
    y_exp = yy[ninst-nsplit-1:]  # create the y experimental dataset for predictions

    feature_info = ""
  #  feature_info = mutual_info_regression(X, y)  #  this can be turned off if you want to look at feature importance
    
#  creat the test and 
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)
    
    # create the scaler objects 
    X_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    
    if (toScale == 0):  # do not scale data, so return the unsclaled Train,Test and Experiment data
       return (X_train,X_test,X_exp,y_train,y_test,y_exp,feature_info,X_scaler,y_scaler)


#  create the scaled Train,Test and Predictio time series for the dependent variable    
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    X_exp_scaled = X_scaler.transform(X_exp)

#  create the scaled Train,Test and Predictio time series for the independent variable  
    y_train_scaled = y_scaler.transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)
    y_exp_scaled = y_scaler.transform(y_exp)

    
    return (X_train_scaled,X_test_scaled,X_exp_scaled,y_train_scaled,y_test_scaled,y_exp_scaled,feature_info,X_scaler,y_scaler)


def svrRegression(X,y,nphases):
    ''' this creates the SVR Regression model and runs the model for all provided phases shifts, provided
    by nphases... the different  coefficients are kind of the standard values used '''
    model = SVR(C=1.0,cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='scale',
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

#  run the model for all phase shifts... temp contains the Train,Teste and Predictions scores for all the phase shifts
    temp,feature_info = phaseShiftNew(model,"SVR",X,y,nphases)
   
    return temp,feature_info


def bayesianRegression(X,y,nphases):
    ''' this creates the Bayesian Ridge Regression model and runs the model for all provided phases shifts, provided
    by nphases... the different  coefficients are kind of the standard values used '''
    model = BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True,
       fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, n_iter=300,
       normalize=False, tol=0.001, verbose=False)

    temp,feature_info = phaseShiftNew(model,"Bayesian",X,y,nphases)

 

    return  temp,feature_info

def linearRegression(X,y,nphases):
    ''' this creates the Linear  Regression model and runs the model for all provided phases shifts, provided
    by nphases '''
    model = LinearRegression()

    temp,feature_info = phaseShiftNew(model,"Linear",X,y,nphases)
   
    return temp,feature_info


def regression(model,model_name,X,y):
    '''  This actuall runs the different regressions models.. it is called by the phaseShiftNew function
    model contains the Regression model that has already been created and model_name contains the 
    name of the model.  This will fit the model to the Test data and predict values using the 
    Experimental/Prediction data and returns various scores, plus the original prediction time series
    plus the acutal predicted values by the model.  Note that the return time series are unscaled
    and should match the original input series. '''

#  train the data
    X_train_scaled,X_test_scaled,X_exp_scaled,y_train_scaled,y_test_scaled,y_exp_scaled,feature_info,X_scaler,y_scaler = train(X,y,24,1)

#  fit the training data 
    model.fit(X_train_scaled, y_train_scaled)

#  Get the Training and Test scores
    train_score = model.score(X_train_scaled,y_train_scaled)
    test_score = model.score(X_test_scaled,y_test_scaled)

#  predict the dependent variables using the independent data 
    y_predict = model.predict(X_exp_scaled)
#  score how well the model predictions are
    r2 = r2_score(y_exp_scaled,y_predict)

#  reverse scale the dependent original and predicted Experimental/Prediction data
    y_predict = y_scaler.inverse_transform(y_predict)
    y_exp_scaled = y_scaler.inverse_transform(y_exp_scaled)
    X_exp_scaled = X_scaler.inverse_transform(X_exp_scaled)

#  the inverse_transform functions return nested Numpy arrays that jsonify cannot handle, so 
#  convert these to Python lists
    bb = []
    for a in y_exp_scaled:
      bb.append(a[0])

    return r2,train_score,test_score,feature_info,bb,y_predict,X_exp_scaled

def phaseShiftNew(model,model_name,X,y,nphases):
    '''  This handles the phase shifting of the dependent and indepenent time series, then 
    calls the regression function to actually perform the Regression, using the model provided
    by the model variable in the input list.  nphases contains the number of phases shifts 
    that should be run '''

#  create temporary lists to peform the phases shifts of the time series 
    tempx = X
    tempy = y
    hold = []

#  this is the primary loop of the function and iterates over all the different phase shifts 
    for nn in range(0,nphases+1):

#  phase shift the dependent variable by removing values from the start of the list
        tempy = y[nn:]
#  the dependent and independent variables each need to have the same number of elements, so 
#  we must removed the last element from the indepedent variables to make the time series match
        if (nn > 0): 
          tempx = X[:0-nn]
        else:
          tempx = X
        
#  now run the Regression for the given model using the time shifted time series
        (r2,train_score,test_score,feature_info,y_exp,y_pred,X_exp) = regression(model,model_name,tempx,tempy)
        temp ={}  #  temporary dict to store the scores for the given phase shift
        temp[nn] = {}
        mm=0

       
        temp[nn]['feature_score'] = {}
        for feat in indices:
    
           temp[nn]['feature_score'][feat] = feature_info[mm]  #  note used right now 
           mm+=1
           
         
        temp[nn]["model"] = model_name  # store the model name
        temp[nn]["r2"] = r2  #  store the score for the predicted ouput
        temp[nn]["train"] = train_score  # score the training data
        temp[nn]["test"] = test_score  #  score for the test data
        
        temp[nn]['y_obs'] = y_exp  #  this is the original dependent data used for the prediction
    #   temp[nn]['y_pred'] = []
        
        
        temp[nn]['y_pred'] = y_pred.tolist()  #  the predicted values by the model
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
    '''  This handles the phase shifting of the dependent and indepenent time series, then 
    calls the classifier function to actually perform the Classification, using the model provided
    by the model variable in the input list.  nphases contains the number of phases shifts 
    that should be run '''
#  create temporary lists to peform the phases shifts of the time series 
    tempx = X
    tempy = y
    hold = []

#  this is the primary loop of the function and iterates over all the different phase shifts 
    for nn in range(0,nphases+1):
#  phase shift the dependent variable by removing values from the start of the list
        tempy = y[nn:]
#  the dependent and independent variables each need to have the same number of elements, so 
#  we must removed the last element from the indepedent variables to make the time series match
        if (nn > 0): 
          tempx = X[:0-nn]
        else:
          tempx = X
        
 #  now run the Regression for the given model using the time shifted time series
        (r2,train_score,test_score,y_pred,y_exp) = classifier(model,model_name,tempx,tempy)
  
        temp ={}  #  temporary dict to store the scores for the given phase shift
        temp[nn] = {}
                   
        temp[nn]["model"] = model_name # store the model name
        temp[nn]["r2"] = r2 #  store the score for the predicted ouput
        temp[nn]["train"] = train_score  # score the training data
        temp[nn]["test"] = test_score #  score for the test data
        
        temp[nn]['y_obs'] = y_exp #  this is the original dependent data used for the prediction
    
        
        temp[nn]['y_pred'] = y_pred #  the predicted values by the model

        hold.append(temp)

    return (hold)

def regLogistic(X,y,nphases):
    ''' this creates the Logistic-regression Classification model and runs the model for all provided phases shifts, provided
    by nphases... '''

    model = LogisticRegression()
    temp = phaseShiftClassify(model,"Logistic",X,y,nphases)

    return temp



def randomForestClassification(X,y,nphases):
    ''' this creates the Random Forest Classification model and runs the model for all provided phases shifts, provided
    by nphases... the different  coefficients are kind of the standard values used '''
   
    model = RandomForestClassifier(n_estimators=200)
    temp = phaseShiftClassify(model,"RFC",X,y,nphases)

    return temp




def svcClassification(X,y,nphases):
    ''' this creates the SVC Classification model and runs the model for all provided phases shifts, provided
    by nphases... the different  coefficients are kind of the standard values used '''
    model = SVC(kernel='linear')

    temp = phaseShiftClassify(model,"SVC",X,y,nphases)

    return temp


def classifier(model,model_name,X,y):
    '''  This actuall runs the different Classification models.. it is called by the phaseShiftClassify function
    model contains the Regression model that has already been created and model_name contains the 
    name of the model.  This will fit the model to the Test data and predict values using the 
    Experimental/Prediction data and returns various scores, plus the original prediction time series
    plus the acutal predicted values by the model.  Note that the return time series are unscaled
    and should match the original input series. '''

#  create the Train,Test nad Experiment/Prediction time series... note that in this scale,
#  the time series are NOT scaled
    X_train_scaled,X_test_scaled,X_exp_scaled,y_train_scaled,y_test_scaled,y_exp_scaled,feature_info,X_scaler,y_scaler = train(X,y,24,0)
    
#  fit the model to the Training data    
    model.fit(X_train_scaled, y_train_scaled)

#  get scores for the Training and Test data
    train_score = model.score(X_train_scaled, y_train_scaled)
    test_score = model.score(X_test_scaled, y_test_scaled)
   
#  create the predicted data by the model   
    y_predict = model.predict(X_exp_scaled)

#  convert Numpy array to a Python list so that jsonify can handle it
    bb = []
    for a in y_predict:
      bb.append(int(a))
    y_predict = bb

#  convert Numpy array to a Python list so that jsonify can handle it
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

#  score the predicted data vs the original dependent data
    for mm in range(0,len(y_predict)):
        if (y_predict[mm] == y_exp_scaled[mm]):
            good+=1
        else:
            bad+=1
           
   
    pp =  good / (good+bad)  #  the score 

    return pp,train_score,test_score,y_predict,y_exp_scaled



@app.route("/classify/<city>/<variable>/<ind>/<classType>/<numBins>") #/getdata/___/___
def classify(city,variable,ind,classType,numBins):
    ''' this function handles the input data from the web page and calls the different
    Classification models '''

#  set flag for whether user wants the standard-deviation based bins created by me or 
#  has input the number of bins they want
    if (classType == "PBIN"):
       ctype = 1
    elif (classType == "CBIN"):
       ctype = 2


  #  split the inices input string into a individual values
    indices = ind.split(",")
    nphases = 12  # set the number of phase shifts to consider

#  get the climate data for the given city from MongoDB    
    mydoc = mycol.find({"city":city})
    data = pd.DataFrame()
# convert year-month pairs from string to integers
    features['yearmo'] = features['yearmo'].astype(int)
#  process the returned data form MongoDB and extract the requested variable
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
#  compute min and max for the data for use in creating bins
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

#  compute the start date and end date for the variable
        start = min(yrmos)
        end = max(yrmos)    
        
        s1 = variable + " MEANS" 
        s2 = variable + " STDEV"   
        data[variable] = temp
        start = min(yrmos)
        end = max(yrmos)
        
#  get request climate indices data and store in Pandas dataframe        
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
#  create bins bases on  less than -3 stdDev, -3 to -2 stdDevs, -2 -> -1 stdDevs, -1 to 1 stdDevs,
# 1 -> 2 stdDevs, 2 -> 3 stdDevs and > 3 stdDevs
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
#  bin data based on # of requests bins
            for key,value in ts.items():
                year = key[0:4]
                mo   = int(key[4:6])
                go=1
                nn=0
                yrmos.append(key)
#  create the scale for each bin range
                scale = (stats[mo]['max'] - stats[mo]['min'])/nbins
                bin=-1
                value = float(value)
#  bin each value 
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
            
#  reshape data for input into different models
    temp = pd.DataFrame(bins)
    y = temp.values.reshape(-1,1)
    
    output = []

#  now run the different classifcation models
    temp = regLogistic(X,y,nphases )
    output.append(temp)  

    temp = svcClassification(X,y,nphases )
    output.append(temp)  

    temp = randomForestClassification(X,y,nphases)
    output.append(temp)


    hld = {}
    hld['yearmo'] = yrmos
   
#   output.append(hld)

    return jsonify(output)  #  return jsonified data as required by Flask


@app.route("/machine/<city>/<variable>/<machine>/<ind>") #/getdata/___/___
def machine(city,variable,machine,ind):
    ''' this function handles the input data from the web page and calls the different
    Regression models '''

#  split the input climate indices string into individual values
    indices = ind.split(",")

# retrieve climate data for the requested city from MongoDB
    mydoc = mycol.find({"city":city})
    data = pd.DataFrame()
#  convert year-month strings to integers
    features['yearmo'] = features['yearmo'].astype(int)
 
#  iterate over the returned data from mongo
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
    
 
    y = data[variable].values.reshape(-1,1)
    temp = pd.DataFrame()
#  get the requested climate indices data
    for val in indices:
      val = val.lower()
      temp[val] = features[val][(features["yearmo"] >= start) & (features['yearmo'] <= end)].astype(float)
    
#  reshape the data for input into the various models
    nl = len(indices)
    X = temp.values.reshape(-1,nl) 
 
#  call the different Regression models
    scores = []
    temp,feature_info = svrRegression(X,y,12)
    scores.append(temp)

    temp,feature_info = linearRegression(X,y,12)
    scores.append(temp)

    temp,feature_info = bayesianRegression(X,y,12)
    scores.append(temp)
    print ("FI ",feature_info)
 


    return jsonify(scores)  #  return jsonified data as required by Flask



@app.route("/neural/<city>/<variable>/<ind>/<phase>/<layers>/<nodes>") #/getdata/___/___
def neural(city,variable,ind,phase,layers,nodes):
    '''  this function handles the input data for the Neural Network and then runs the 
    Neural Network '''
# convert the input data from string to integers
    phase = int(phase) 
    layers = int(layers)
    nodes = int(nodes)
# split input climate indices string to invividual values
    indices = ind.split(",")

#  get the climate data for the requested city from MondoDB
    mydoc = mycol.find({"city":city})
    data = pd.DataFrame()

#  convert year-month string to integers
    features['yearmo'] = features['yearmo'].astype(int)
  #  print (features.info())

# iterate over returned data from MondoDB and extract requested variable
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
    
#  reshaped data for input into neural netowrk
    y = data[variable].values.reshape(-1,1).astype(float)
    temp = pd.DataFrame()

#  get requested climate indices
    for val in indices:
      val = val.lower()
      temp[val] = features[val][(features["yearmo"] >= start) & (features['yearmo'] <= end)].astype(float)
    
    nl = len(indices)
    X = temp.values.reshape(-1,nl).astype(float) 

    scores = []
#  phase shift data the requested amount
    tempy = y[phase:]
 
    if (phase > 0): 
        tempx = X[:0-phase]
    else:
        tempx = X
 #  run Neural Network 
    scores = neuralNet(tempx,tempy,layers,nodes)
 

    return jsonify(scores)




def neuralNet(X,y,layers,nodes):
    ''' This runs the Regression Neural Network using the input # of layers and nodes '''
#  train the data... note the data is scaled as this is a Regression 
    X_train_scaled,X_test_scaled,X_exp_scaled,y_train_scaled,y_test_scaled,y_exp_scaled,feature_info,X_scaler,y_scaler = train(X,y,24,1)

 
    nd = X.shape
    ndims = nd[1]

#  create Neural Network Modeladn the number of nodes for each layer
    model = Sequential()

#  add layers to the network 
    for nn in range(1,layers):
      print ("Adding Layer ",nn)
      model.add(Dense(units=nodes, activation='relu', input_dim=ndims))

#  add the ouput layers
    model.add(Dense(units=1))
    
#  compile the model
    model.compile(optimizer='adam',
                loss='mse',
                metrics=['mse'])

   
#  fit the model, use 100 epochs, but at some point, let users input the number of epochs
    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        epochs=100,
        shuffle=True,
        verbose=0,
        validation_split = 0.2
    )
    hist = pd.DataFrame(history.history)

    model.summary()

# predict the dependent values for the for Experiment/Prediction time series
    y_predict = model.predict(X_exp_scaled)
    
#  score the Experiment data
    r2 = model.evaluate(X_exp_scaled,y_exp_scaled)

    
    mde=0

#  unscale the data 
    y_predict = y_scaler.inverse_transform(y_predict)
    y_exp_scaled = y_scaler.inverse_transform(y_exp_scaled)

#  create the error score so that we now exactly how well the predictions worked
    for nn in range(0,len(y_exp_scaled)):
        mde+= (y_exp_scaled[nn][0]-y_predict[nn][0])**2
        
    mde=math.sqrt(mde/len(y_exp_scaled))
    
   
#  convert the different Numpy arrays to Python lists so that jsonify can handle them
    bb = []
    hold = []
    for a in y_exp_scaled:
      bb.append(float(a[0]))

    cc = []
    for a in y_predict:
      cc.append(float(a[0]))
    
    temp = {}
    temp["r2"] = mde  #  my created MDE score 
    
    temp['y_obs'] = bb  # original data for precition comparison
    temp['y_pred'] = cc # the predicted values from the model
     

#  convert the Numpy arrays to Python lists
    temp['mse'] = hist['mean_squared_error'].values.tolist()
    temp['mse_val'] = hist['val_mean_squared_error'].values.tolist()

    hold.append(temp)

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
       


if __name__ == "__main__":  
    app.run(debug=True)
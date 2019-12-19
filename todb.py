import pandas as pd 
import numpy as np
import csv
import sqlite3
import sqlalchemy as db
import pymongo
import statistics

#myclient = pymongo.MongoClient("mongodb://localhost:27017/")
#mydb = myclient["climate"]
#mycol = mydb["monthly"]

client = pymongo.MongoClient("mongodb+srv://jgirlsdad:444jayla@cluster0-dgjk9.mongodb.net/test?retryWrites=true&w=majority")
mydb = client.climate
mycol = mydb["monthly"]

def getMean(dd):
    number_months = []
    number_months = [0 for nn in range(0,13)]
    num = [0 for nn in range(1950,2020)]
    means = [0 for nn in range(0,13)]
    sdevs = [0 for nn in range(0,13)]
    temp = {}
    
    for date in dd:
        spl = date.split("-")
        
        year = int(spl[0])
        month = int(spl[1])
        num[year-1950]+=1
        if (year >= 1950 and year <= 2018):
          try:
             val = float(dd[date])
             means[int(spl[1])]+= float(dd[date])
             
          except:
            print ("Bad digit ",date,dd[date])
            
          
          if month in temp:
              temp[month].append(val)
          else:
              temp[month] = []
              temp[month].append(val)
          number_months[int(spl[1])]+=1
   
    for nn in range(1,len(means)):       
        means[nn] = means[nn]/number_months[nn]
        sdevs[nn] = statistics.stdev(temp[nn])
        print (nn)
    
    return means[1:],sdevs[1:]


with open("data/US-cities-1.csv") as csvfile:
    reader = csv.reader(csvfile)
    row = next(reader)
    row = next(reader)
    stn0 = row[0]
    city = row[1]
    lat = row[2]
    lon = row[3]
    start = row[5]
    csvfile.seek(0)
    row = next(reader)
    tavg = {}
    prcp = {}
    tmax = {}
    tmin = {}
    hist={}
    print (city)
    for row in reader:
        if (stn0 != row[0]):
          
                # record['name'] = name
                # xx = mycol.insert_one(record)
          record = {}
          record['city'] = city
          record['lat'] = lat
          record['lon'] = lon
          record['start'] = start
          record['end'] = end
          record['TAVG'] = tavg
          record['PRCP'] = prcp
          record['TMIN'] = tmin
          record['TMAX'] = tmax
          
          
          record['TAVG MEANS'],record['TAVG STDEV']  = getMean(tavg)
          record['PRCP MEANS'],record['PRCP STDEV']  = getMean(prcp)
          record['TMIN MEANS'],record['TMIN STDEV']  = getMean(tmin)
          record['TMAX MEANS'],record['TMAX STDEV']  = getMean(tmax)
         
          xx = mycol.insert_one(record)
          tavg = {}
          prcp = {}
          tmax = {}
          tmin = {}
          start = row[5]
          print (row[1])
          for yy,cnt in hist.items():
              if cnt < 12:
                  print (yy,cnt)
        
        spl = row[5].split("-")
        year = int(spl[0])
        if (year >= 1950 and year <= 2018):
            tavg[row[5]] = row[7]
            prcp[row[5]] = row[6]
            tmin[row[5]] = row[8]
            tmax[row[5]] = row[9]
            city = row[1]
            lat = row[2]
            lon = row[3]
            end = row[5]
            if year in hist:
                hist[year]+=1
            else:
                hist[year]=1
        stn0=row[0]

    record = {}
    record['city'] = city
    record['lat'] = lat
    record['lon'] = lon
    record['start'] = start
    record['end'] = end
    record['TAVG'] = tavg
    record['PRCP'] = prcp
    record['TMIN'] = tmin
    record['TMAX'] = tmax
    

    record['TAVG MEANS'],record['TAVG STDEV']  = getMean(tavg)
    record['PRCP MEANS'],record['PRCP STDEV']  = getMean(prcp)
    record['TMIN MEANS'],record['TMIN STDEV']  = getMean(tmin)
    record['TMAX MEANS'],record['TMAX STDEV']  = getMean(tmax)
    xx = mycol.insert_one(record)

    for yy,cnt in hist.items():
        if cnt < 12:
            print (yy,cnt)


def getCat(val,mean,std):

   if val > (mean+std):
        return 1
   elif val < (mean-std):
        return -1
   else:
        return 0
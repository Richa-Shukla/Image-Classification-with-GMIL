
import fiona
import random
import heapq as hp
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.special import kl_div
from osgeo import gdal

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

shape = fiona.open('D:/Documents/train data.shp')
driver=gdal.GetDriverByName('GTiff')
filename="C:/Users/Dell User/Desktop/Automated Learning and Data Analysis/Project/Gaussian/GMIL-Data/image.tif"
dataset=gdal.Open(filename)
band1=dataset.GetRasterBand(1)
band2=dataset.GetRasterBand(2)
band3=dataset.GetRasterBand(3)
cols=dataset.RasterXSize
rows=dataset.RasterYSize
transform=dataset.GetGeoTransform()
xorigin=transform[0]
yorigin=transform[3]
pixelwidth=transform[1]
pixelheight=-transform[5]
data=band1.ReadAsArray(0,0,cols,rows)
count=0
print (shape.schema)
p=[]

def KLDivergence(cov_p,cov_q,mean_p,mean_q):
   print(cov_q)
   cov_q_inv = np.linalg.inv(cov_q)
   prod = cov_q_inv*cov_p
   trace_to = np.trace(prod)
   det_p = np.linalg.det(cov_p)
   det_q = np.linalg.det(cov_q)
   mean_diff = np.subtract(mean_p,mean_q)
   mean_trans = np.transpose(mean_diff)
   #print(np.mean_trans)
   distance_KLD = 1/2* ((np.log2(det_q) - np.log2(det_p))+trace_to + np.matmul(np.matmul(mean_trans,cov_q_inv),mean_diff))
   return distance_KLD.round()
def knn(mean_q,cov_q,k,data):
    #print('parami')
    ##Majority voting
    count=[0]*7
    h=[]*k
    data=json.loads(json.dumps(data))
    for p in data['parameters']:
        #print(p["mean"])
        #print('Mean: ' + p['mean'])
        #print('Covariance: ',p["covariance"])
        #print('Class: ' ,p["classlabel"])
        #print('') 
        cov_p=np.array(p["covariance"])
        mean_p=p["mean"]
        class_p=p["classlabel"]
        kpq=KLDivergence(cov_p,cov_q,mean_p,mean_q)
        kqp=KLDivergence(cov_q,cov_p,mean_q,mean_p)
        kldivergence=0.5*(kpq+kqp)
        if len(h)<=k:
         hp.heappush(h,[kldivergence,class_p])
        else:
         if kldivergence<np.any(max(h)):     
          index=h.index(max(h))
          h[index]=[kldivergence,class_p]
          #print(h)
    
    for p in h:
      #print('cont is',p[1])
      count[p[1]-1]=count[p[1]-1]+1
    print('class label of the test data is',count.index(max(count))+1)
    return count.index(max(count))+1
      #print(count)

        
                    
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def getPatches(row,col):
    if np.array(band1.ReadAsArray(row,col,10,10)).size < 10:
        return np.zeros((10,10)),np.zeros((10,10)),np.zeros((10,10))
    patch1=np.array(band1.ReadAsArray(row,col,10,10)).reshape(100)
    if np.array(band2.ReadAsArray(row,col,10,10)).size < 10:
        return np.zeros((10,10)),np.zeros((10,10)),np.zeros((10,10))
    patch2=np.array(band2.ReadAsArray(row,col,10,10)).reshape(100)
    if np.array(band3.ReadAsArray(row,col,10,10)).size < 10:
        return np.zeros((10,10)),np.zeros((10,10)),np.zeros((10,10))
    patch3=np.array(band3.ReadAsArray(row,col,10,10)).reshape(100)
    return patch1,patch2,patch3
    
def getCovariance(row, col):
    
    return np.cov(np.vstack(getPatches(row,col)))
    
def getMean(row,col):
    patch1,patch2,patch3 = getPatches(row,col)
    return [np.mean(patch1),np.mean(patch2),np.mean(patch3)]
cov=[]
means=[]
count=0
#data1=[]
test={}
train={}
test['parameters']=[]
train['parameters']=[]
data={}
#train=[]
#train['data']=[]
dictlist = [dict() for x in range(10)]
#train=dict(train)
data['parameters']=[]
with open('data.json','w')as outfile:
 for a in shape:
    temp=a['geometry']['coordinates']
    cls=a['properties']['class']
    #if count==0:
       #print('class of 0 is',cls)
    if cls==None or cls==0:
        continue
    #print('class of 0 is',cls)
    col=int((temp[0]-xorigin)/pixelwidth)
    row=int((yorigin-temp[1])/pixelheight)
    
    covarience = getCovariance(row,col)
    mean = getMean(row,col)
    #p1=np.vstack(getPatches(row,col))
    #p2=np.vstack(getPatches(row,col))
    #print(kl_div(p1,p2))
    if(np.any(covarience==0)):
        continue
        
    c=covarience.tolist()
    #co=covarience.reshape(9)
    #m=mean.tolist()
    if count % 10==0:
     test['parameters'].append({'mean':mean,'covariance': c,'classlabel': cls})
    else:
     train['parameters'].append({'mean':mean,'covariance': c,'classlabel': cls})
    #if count!=0:
     #dictlist[count]={'mean':mean,'covariance': c,'classlabel':cls}
     #print(dictlist[count]['mean'])
     #train['data'].append({'mean':mean,'covariance': c,'classlabel':cls})
    json.dumps(train)
    json.dumps(test)
    #train=json.loads(json.dumps())
    #json.dump(data,outfile,cls=NumpyEncoder)
    #print(covarience)
    #print(mean)
    count=count+1
    cov.append(covarience)
    means.append(mean)
    #if count==10:
    #   break
'''with open('data.json') as json_file:  
    #data = json.loads(json_file)
    data=json.loads(json.dumps(data))
    for p in data['parameters']:
        print(p["mean"])
        #print('Mean: ' + p['mean'])
        print('Covariance: ',p["covariance"])
        print('Class: ' ,p["classlabel"])
        print('')    '''
#print(KLDivergence(cov[0],cov[0],means[0],means[0]))
#train=[[means[1],cov[1],classmeans[2]]]
#knn(means[1],cov[1],1,cov[0],means[0],1)
correct=0
wrong=0
print("HELLLLLLLLLLLLLLLLLLLLLLLLLLO")
train=json.loads(json.dumps(train))
test=json.loads(json.dumps(test))
for p in test['parameters']:
 cov_p=np.array(p["covariance"])
 mean_p=p["mean"]
 class_p=p["classlabel"]
 label=knn(mean_p,cov_p,30,train)
 if label==class_p:
   correct=correct+1
 else:
   wrong=wrong+1
print(correct,wrong)
 

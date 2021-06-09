```python
# 數據處理  Pima Indians Diabetes Database  資料集
# 資料來源 ( Kaggle ) https://www.kaggle.com/saurabh00007/diabetescsv
# 儲存在本地相對目錄:  './DataSet/Diabetes/diabetes.csv'
# 實作: 參考 https://www.youtube.com/watch?v=JDU3AzH3WKg

# 讀取資料 
FileName = r'./DataSet/Diabetes/diabetes.csv'
```


```python
# numpy 讀取 CSV 檔
# genfromtxt 函數

import numpy as np
from numpy import genfromtxt

my_data = genfromtxt(FileName,dtype=float,delimiter=',')
my_data

```




    array([[    nan,     nan,     nan, ...,     nan,     nan,     nan],
           [  6.   , 148.   ,  72.   , ...,   0.627,  50.   ,   1.   ],
           [  1.   ,  85.   ,  66.   , ...,   0.351,  31.   ,   0.   ],
           ...,
           [  5.   , 121.   ,  72.   , ...,   0.245,  30.   ,   0.   ],
           [  1.   , 126.   ,  60.   , ...,   0.349,  47.   ,   1.   ],
           [  1.   ,  93.   ,  70.   , ...,   0.315,  23.   ,   0.   ]])




```python
# pandas 讀取 CSV 檔
# read_csv 函數

import pandas as pd

df = pd.read_csv(FileName)
print('檢視資料:\n',df.head(n=3),'\n')


# 習慣偏好使用
print('簡易敘述統計:\n',df.describe(),'\n')

print('檢查闕漏值:\n',df.isnull().count(axis=0),'\n')

print('資料欄位名稱:\n',df.columns,'\n')
# 參考資料: https://zhuanlan.zhihu.com/p/90912671
```

    檢視資料:
        Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \
    0            6      148             72             35        0  33.6   
    1            1       85             66             29        0  26.6   
    2            8      183             64              0        0  23.3   
    
       DiabetesPedigreeFunction  Age  Outcome  
    0                     0.627   50        1  
    1                     0.351   31        0  
    2                     0.672   32        1   
    
    簡易敘述統計:
            Pregnancies     Glucose  BloodPressure  SkinThickness     Insulin  \
    count   768.000000  768.000000     768.000000     768.000000  768.000000   
    mean      3.845052  120.894531      69.105469      20.536458   79.799479   
    std       3.369578   31.972618      19.355807      15.952218  115.244002   
    min       0.000000    0.000000       0.000000       0.000000    0.000000   
    25%       1.000000   99.000000      62.000000       0.000000    0.000000   
    50%       3.000000  117.000000      72.000000      23.000000   30.500000   
    75%       6.000000  140.250000      80.000000      32.000000  127.250000   
    max      17.000000  199.000000     122.000000      99.000000  846.000000   
    
                  BMI  DiabetesPedigreeFunction         Age     Outcome  
    count  768.000000                768.000000  768.000000  768.000000  
    mean    31.992578                  0.471876   33.240885    0.348958  
    std      7.884160                  0.331329   11.760232    0.476951  
    min      0.000000                  0.078000   21.000000    0.000000  
    25%     27.300000                  0.243750   24.000000    0.000000  
    50%     32.000000                  0.372500   29.000000    0.000000  
    75%     36.600000                  0.626250   41.000000    1.000000  
    max     67.100000                  2.420000   81.000000    1.000000   
    
    檢查闕漏值:
     Pregnancies                 768
    Glucose                     768
    BloodPressure               768
    SkinThickness               768
    Insulin                     768
    BMI                         768
    DiabetesPedigreeFunction    768
    Age                         768
    Outcome                     768
    dtype: int64 
    
    資料欄位名稱:
     Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
           'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
          dtype='object') 
    
    


```python


# pima-indians-diabetes.csv 糖尿病數據集
def loadData(fileName):
    dataList = []; labelList = []
    fr = open(fileName,'r')
    
    line = fr.readlines()
    ind = 0
    for item in line:        
        
        if ind == 0:
            header = item.strip().split(',')
        else:
            curLine = item.strip().split(',')
            #print(curLine[0:-1],curLine[-1])
            dataList.append([ float(num) for num in curLine[0:-1]])
            labelList.append(float(curLine[-1]))
            pass
        ind += 1
        
    return np.array(dataList), np.array(labelList),header

samples, labes, header =loadData(FileName)



```


```python

x ,y = samples[10:] , labes[10:]

def ModelFitFunction(x,y,Rat=0.001,Epoch=2):
    N,F = x.shape
#     np.random.seed(0)
#     W,B = np.random.randn(F),np.random.randn(1)

    np.random.seed(0)
    W,B = np.zeros(F),0
    for _ in range(Epoch):
        #print('Epoch {} '.format(_))
        linear = np.dot(x,W) +B
        yhead = 1.0 /(1.0 + np.exp(-linear))
        dW= (1/x.shape[0])*np.dot(x.T, (yhead-y))
        dB= (1/x.shape[0])*np.sum(yhead-y)
        
        W -= Rat * dW
        B -= Rat * dB
    
    return W,B

Weight,Bias = ModelFitFunction(x*0.001,y,Rat=0.001,Epoch=10000)
print(Weight,Bias)


test_x ,test_y = samples[0:20] , labes[0:20]
linear = np.dot(test_x*0.001,W) +B
yhead = 1.0 /(1.0 + np.exp(-linear))
yhead_class = [1 if i > 0.5 else 0 for i in yhead]

print(yhead)
print(yhead_class)
print(test_y)
print(test_x[0])
Total = len(yhead_class)
right = 0
for _ in range(Total):
    if yhead_class[_] == test_y[_]:
        right += 1
        
print('acc',right/Total)

    
    

```

    [ 1.38221054e-03  1.14222678e-03 -3.37296798e-02 -6.36918367e-03
      2.30813874e-02 -7.03411488e-03 -9.11990577e-06 -6.02853000e-03] -0.5735671801278932
    [0.51618982 0.50605999 0.50214662 0.54683029 0.58145676 0.4965996
     0.54426755 0.47885469 0.75247088 0.50927335 0.49690392 0.50029567
     0.50107584 0.8340026  0.5910187  0.47746396 0.62294986 0.495657
     0.53858347 0.55288727]
    [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1]
    [1. 0. 1. 0. 1. 0. 1. 0. 1. 1. 0. 1. 0. 1. 1. 1. 1. 1. 0. 1.]
    [  6.    148.     72.     35.      0.     33.6     0.627  50.   ]
    acc 0.7
    

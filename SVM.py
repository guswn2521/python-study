#!/usr/bin/env python
# coding: utf-8

# ### Support Vector Machine
# : 요소를 최적으로 분류하는 구분선을 찾는 방식
# - 구분선의 매개변수를 조정해 margin이 최대인 선
# - support 서포트 벡터 : 선과 가장 가까운 데이터
# - margin : 선과 가장 가까운 양 옆 데이터와의 거리 = 선과 서포트 벡터와의 거리
# - decision boundary : 데이터 구분선
# - robustness : outlier의 영향을 받지 않는 정도
#     - 평균 : robust하지 않음
#     - 중앙값 : robust
# - 커널 트릭 : 데이터를 고차원 공간에 매핑하여 구분선을 찾는 방법
#     - z = x^2 + y^2
# - c : 구분선이 smooth한지 classifying한지 나타내는 지표
#     - 크면 더 정확하게 구분, 굴곡짐
#     - 작으면 smooth, 직선에 가까움, 잘 구분못함.
# - gamma(Y) : 데이터 하나가 구분선에 영향을 주는 범위
#     - reach : 구분선에 영향을 주는 데이터 범위
#     - gamma가 작으면 reach가 멀다 -> 멀리있는 데이터까지 구분선에 영향이 있음 => 가까운 데이터의 영향이 상대적으로 적어져 직선에 가까워짐
#     - gamma가 크면 reach가 작다 -> 가까이 있는 데이터만 구분선에 영향이 있음 => 가까운 데이터의 영향이 상대적으로 커져 굴곡짐
# - C는 클수록 데이터를 정확하게 구분하려함
# - gamma는 클수록 개별 데이터마다 구분선을 만들려 함

# In[123]:


import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split


# In[2]:


## bmi.csv 데이터 만들기
def calc_bmi(h,w):
    bmi = w/(h/100)**2
    if bmi < 18.5: return 'thin'
    if bmi < 25: return 'normal'
    return 'fat'

## 파일 쓰기 : 같은이름 파일이면 덮어씀.
fp = open('bmi.csv','w',encoding='utf-8')
fp.write('height,weight,label\r\n')
cnt = {'thin':0, 'normal':0, 'fat':0}
for i in range(20000):
    h = random.randint(120,200)
    w = random.randint(35, 80)
    label = calc_bmi(h,w)
    cnt[label] +=1
    fp.write(f"{h},{w},{label}\r\n")
fp.close()
print('ok',cnt)


# In[45]:


tbl = pd.read_csv('bmi.csv')
tbl.head()


# In[74]:


## series
## w, h 0과 1사이로
label = tbl['label']
w = tbl['weight'] / 120
h = tbl['height'] / 210
wh = pd.concat([w, h], axis=1)


# In[52]:


## test, train data 나누기
data_train, data_test, label_train, label_test =     train_test_split(wh, label)


# In[53]:


## 데이터 훈련 모델만들기
clf = svm.SVC()
clf.fit(data_train, label_train)

predict = clf.predict(data_test)

ac_score = metrics.accuracy_score(label_test, predict)
cl_report = metrics.classification_report(label_test, predict)
print(ac_score)
print(cl_report)


# In[79]:


## 그래프 그리기
tbl = pd.read_csv('bmi.csv', index_col=2)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

def scatter(lbl, color):
    b = tbl.loc[lbl]
    ax.scatter(b['weight'],b['height'], c=color, label=lbl)
scatter('fat', 'red')
scatter('normal', 'yellow')
scatter('thin','purple')
ax.legend()


# In[115]:


tbl = pd.read_csv('bmi.csv')
label = tbl['label']
w = tbl['weight'] / 120
h = tbl['height'] / 210
wh = pd.concat([w,h], axis=1)

data_train, data_test, label_train, label_test = train_test_split(wh, label)

clf = svm.LinearSVC()
clf.fit(data_train, label_train)
predict = clf.predict(data_test)
ac_score = metrics.accuracy_score(label_test, predict)
cl_report = metrics.classification_report(label_test, predict)
print(ac_score)
print(cl_report)


# In[138]:


from pydataset import data
iris = data('iris')
iris


# In[153]:


iris_sp = iris[ (iris['Species']=='setosa') | (iris['Species']=='versicolor')]


# In[192]:


iris_datasets = iris_sp.drop(columns=['Petal.Length','Petal.Width'])


# In[203]:


iris_datasets.index = iris_datasets['Species']

iris_datasets.drop(columns='Species', inplace=True)


# In[220]:


iris_datasets.head()


# In[221]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


# In[223]:


scaler.fit(iris_datasets)

iris_scaled = scaler.transform(iris_datasets)


# In[225]:


iris_nor = pd.DataFrame(data=iris_scaled, columns=['sepal length','sepal width'])
iris_nor.head()


# In[229]:


data_train, data_test, label_train, label_test = train_test_split(iris_nor, iris_datasets.index)


# In[233]:


from sklearn.svm import SVC

model = SVC(kernel='linear', C=1e10).fit(data_train, label_train)

prediction = clf.predict(data_test)
prediction


# In[322]:


label_pre = pd.Series(prediction)

coloring = {'setosa':'purple', 'versicolor':'yellow'}

plt.figure()
plt.scatter(data_test.iloc[:,0], data_test.iloc[:,1], c=label_pre.apply(lambda x: coloring[x]))


# In[271]:


label = pd.Series(label_train,name='Species')

graph_df = pd.concat([data_train,label],axis=1)

df = iris2.iloc[:,[0,1,4]]


# In[319]:


label_train2 = pd.Series(label_train)

colors = {'setosa':'red', 'versicolor':'blue'}

plt.figure()
y = data_train['sepal length']
x = data_train['sepal width']
plt.scatter(x,y,c=label_train2.apply(lambda x: colors[x]))
# plt.scatter(label_test,aes=(x,y, c=prediction))


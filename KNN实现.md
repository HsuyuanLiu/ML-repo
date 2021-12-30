# KNN实现

### 基本内容

k近邻法（k-nearest neighbor）是一种基本分类与回归方法。其输入为实例的特征向量，对应于特征空间的点；输出为实例的类别，可以取多类。**k近邻法假设给定一个训练数据集，其中的实例类别已定。分类时，对新的实例，根据其k个最近邻的训练实例的类别，通过多数表决等方式进行预测。**其实质是利用训练集中的数据，对特征向量空间进行划分，并作为其分类的模型。

模型由三个基本要素——距离度量、k值的选择和分类决策规则决定。

在本次实验中，选择曼哈顿距离作为距离度量的基本依据，针对多个K值进行实验，完成对KNN模型的基本构造过程：

### 代码实现

在数据的预处理阶段，完成对数据集的划分工作：

```python
def load_txtdata(path):
    gross_data=np.loadtxt(path)
    np.random.shuffle(gross_data)
    train_data=[]
    train_label=[]
    for item in gross_data:
        train_data.append(np.asarray(item[0:-10]))
        train_label.append(item[-10:])
    return train_data,train_label 
batch_size=int(0.1*len(data))
def make_dataset(num):
    if num+batch_size<len(data)-1:
        lis=np.arange(num)
        collected=np.arange(num+batch_size,len(data)-1)
        lis=np.append(lis,collected)
    else:
        lis=np.arange(num)
    train_set=[]
    test_set=[]
    test_label=[]
    train_label=[]
    for number in lis:
        train_set.append(data[number])
        train_label.append(labels[number])
    for i in np.arange(num,min(num+batch_size,len(data)-1)):
        test_label.append(labels[i])
        test_set.append(data[i])
    write_data(train_set, train_label,test_set,test_label)
    return train_set, train_label,test_set,test_label
```

在这一过程中，以10%的数据作为单一批次的测试集，使用其它数据作为对测试集进行分类的依据：

而后实现KNN算法：

```python
from collections import Counter
import pandas as pd
def knn(input_num, dataSet,labelset, k):
    store ={}
    for num_set in range(len(dataSet)):
        dist=calculate_distance(input_num,dataSet[num_set])
        store[num_set]=dist
    dict1=sorted(store.items(),key=lambda x:x[1])
    dict2=dict1[0:k]
    vote=[num[0] for num in dict2]

    voted=[]
    for i in vote:
        c=getnumber(labelset[i])
        voted.append(c)
    maxlable=max(voted,key=voted.count)
    return maxlable
```

并对不同的K值进行测试,并与Scikit-learn中的KNN实现进行对比：

```python
def train_test(num,K):
    train_set, train_label,test_set,test_label=make_dataset(num)
    times=0
    accuracy=0
    for i in np.arange(num,min(num+batch_size,len(data)-1)):
      predicted=knn(data[i],train_set,train_label,20)
      labeled=getnumber(labels[i])
      times=times+1
      if(labeled==predicted):
        accuracy=accuracy+1
    return accuracy/times 
config=[1,2,3,5,8,10,20]
average=[]
for item in config:
    average.append
    sum_k=0
    for i in range(5):
        sum_k=sum_k+train_test(np.random.randint(len(data)),item)
    sum_k=sum_k/5
    average.append(sum_k)
print(average)
from sklearn.neighbors import KNeighborsClassifier
train_set, train_label,test_set,test_label=make_dataset(np.random.randint(len(data)))
statistic=[]
for K in config:
    neigh = KNeighborsClassifier(n_neighbors=K)
    reset_set,conv_test=[],[]
    for item in train_label:
        reset_set.append(getnumber(item))
    for item in test_label:
        conv_test.append(getnumber(item))
    neigh.fit(train_set, reset_set)
    acc=0
    predicted=neigh.predict(test_set)
    for item in range(len(predicted)):
        if predicted[item]==conv_test[item]:
            acc=acc+1
    statistic.append(acc/len(predicted))
print(statistic)
```

经过重复实验，可以得到的结果如下;

<img src="C:/Users/32068/Desktop/f06252b3-e22a-45b5-8b6c-3303cef94e60.png" alt="f06252b3-e22a-45b5-8b6c-3303cef94e60" style="zoom:80%;" />

由此结果可以看出，其中对于手动实现的KNN算法，当K=8时取得的效果最好，而整体的结果略优于Scikit-learn包中实现的算法，验证了手写算法的正确性。

### 总结

在本次实验中，完成了实验的基本，中级，提高要求，寻找到了针对这一数据集效果最佳的K值，并完成了与Scikit-learn的对比，进一步熟悉了这一算法以及包中API，也对机器学习任务有了最基本的认识.
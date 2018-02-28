## 1. 常见的算法：

### Microsoft Azure中的主要算法
![Azure_ml_map](https://raw.githubusercontent.com/OnlyBelter/jupyter-note/master/machine_learning/outline/Microsoft_Azure_ML.png)

### scikit-learn中算法的分类
![sk-learn_ml_map](https://raw.githubusercontent.com/OnlyBelter/jupyter-note/master/machine_learning/outline/scikit-learn_ml_map.png)

### 1.1 线性回归
- 岭回归
- Lasso回归

### 1.2 逻辑回归
- Binary Logistic Regression
- Multi-nominal Logistic Regression
- Ordinal Logistic Regression

### 1.3 SVM / SVR
- Linear SVM
- Non-Linear SVM

### 1.4 EM

### 1.5 KNN

### 1.6 PCA

### 1.7 KMeans

### 1.8 朴素贝叶斯
Sentiment Analysis / Document Categorization / Email Spam Filtering

### 1.9 极大似然估计

### 1.10 决策树: 
- 分类回归树CART
- a link for introduction of [decision tree](http://www.cs.princeton.edu/~schapire/talks/picasso-minicourse.pdf) 
- classification trees: These are considered as the default kind of decision trees used to separate a dataset into different classes, based on the response variable. These are generally used when the response variable is categorical in nature.
- regression trees: When the response or target variable is continuous or numerical, regression trees are used.

A decision tree is a graphical representation that makes use of branching methodology to exemplify all possible outcomes of a decision, based on certain conditions. In a decision tree, the internal node represents a test on the attribute, each branch of the tree represents the outcome of the test and the leaf node represents a particular class label i.e. the decision made after computing all of the attributes. The classification rules are represented through the path from root to the leaf node.

### 1.11 神经网络

### 1.12 随机森林
- application
  - Random Forest algorithms are used by banks to predict if a loan applicant is a likely high risk.
  - They are used in the automobile industry to predict the failure or breakdown of a mechanical part.
  - These algorithms are used in the healthcare industry to predict if a patient is likely to develop a chronic disease or not.
  - They can also be used for regression tasks like predicting the average number of social media shares and performance scores.
  - Recently, the algorithm has also made way into predicting patterns in speech recognition software and classifying images and texts.
  
### 1.13 强化学习

### 1.14 高斯生成模型(Gaussian Generative Models)

### 1.15 层次聚类

### 1.16 先验算法
Apriori algorithm is an unsupervised machine learning algorithm that generates association rules from a given data set. Association rule implies that if an item A occurs, then item B also occurs with a certain probability. Most of the association rules generated are in the IF_THEN format.
- [wiki](https://en.wikipedia.org/wiki/Apriori_algorithm)

### 1.17 AdaBoost(Adaptive Boosting)
- [wiki](https://en.wikipedia.org/wiki/AdaBoost)


## 2. 算法的分类：

### 2.1 有监督学习(Supervised learning)
Supervised learning algorithms make predictions based on a set of examples. For instance, historical stock prices can be used to hazard guesses at future prices. Each example used for training is labeled with the value of interest—in this case the stock price. A supervised learning algorithm looks for patterns in those value labels. It can use any information that might be relevant—the day of the week, the season, the company's financial data, the type of industry, the presence of disruptive geopolitical events—and each algorithm looks for different types of patterns. After the algorithm has found the best pattern it can, it uses that pattern to make predictions for unlabeled testing data—tomorrow's prices.

- **分类问题(Classification)**: When the data are being used to predict a category, supervised learning is also called classification. This is the case when assigning an image as a picture of either a 'cat' or a 'dog'. (相当于定性问题)
  - **二分类(two-class or binomial classification)**: there are only two choices
  - **多分类(multi-class classification)**: there are more categories
- **回归问题(Regression)**: When a value is being predicted, as with stock prices, supervised learning is called regression. (相当于定量问题)
- **异常检测(Anomaly detection)**: Sometimes the goal is to identify data points that are simply unusual. In fraud detection, for example, any highly unusual credit card spending patterns are suspect. *The possible variations are so numerous and the training examples so few, that it's not feasible to learn what fraudulent activity looks like. The approach that anomaly detection takes is to simply learn what normal activity looks like (using a history non-fraudulent transactions) and identify anything that is significantly different.*

### 2.2 无监督学习(Unsupervised learning)
In unsupervised learning, data points have no labels associated with them. Instead, the goal of an unsupervised learning algorithm is to organize the data in some way or to describe its structure. This can mean grouping it into clusters or finding different ways of looking at complex data so that it appears simpler or more organized.

### 2.3 强化学习(Reinforcement learning)
In reinforcement learning, the algorithm gets to choose an action in response to each data point. The learning algorithm also receives a reward signal a short time later, indicating how good the decision was. Based on this, the algorithm modifies its strategy in order to achieve the highest reward. Reinforcement learning is common in robotics, where the set of sensor readings at one point in time is a data point, and the algorithm must choose the robot's next action. It is also a natural fit for Internet of Things applications.


## 3. 算法的选择

### 3.1 准确度(Accuracy)
Getting the most accurate answer possible isn't always necessary. Sometimes an approximation is adequate, depending on what you want to use it for. If that's the case, you may be able to cut your processing time dramatically by sticking with more approximate methods. Another advantage of more approximate methods is that they naturally tend to avoid overfitting.

### 3.2 训练时间(Training time)
The number of minutes or hours necessary to train a model varies a great deal between algorithms. Training time is often closely tied to accuracy—one typically accompanies the other. In addition, some algorithms are more sensitive to the number of data points than others. When time is limited it can drive the choice of algorithm, especially when the data set is large.

### 3.3 线性度(Linearity)
Lots of machine learning algorithms make use of linearity. Linear classification algorithms assume that classes can be separated by a straight line (or its higher-dimensional analog). These include logistic regression and support vector machines. Linear regression algorithms assume that data trends follow a straight line. These assumptions aren't bad for some problems, but on others they bring accuracy down.

Despite their dangers, linear algorithms are very popular as a first line of attack. They tend to be algorithmically simple and fast to train.

### 3.4 参数的数量(Number of parameters)
Parameters are the knobs a data scientist gets to turn when setting up an algorithm. They are numbers that affect the algorithm's behavior, such as error tolerance or number of iterations, or options between variants of how the algorithm behaves. The training time and accuracy of the algorithm can sometimes be quite sensitive to getting just the right settings. Typically, algorithms with large numbers parameters require the most trial and error to find a good combination.

The upside is that *having many parameters typically indicates that an algorithm has greater flexibility*. It can often achieve very good accuracy. Provided you can find the right combination of parameter settings.

### 3.5 特征的数量(Number of features)
For certain types of data, the number of features can be very large compared to the number of data points. This is often the case with genetics or textual data. The large number of features can bog down some learning algorithms, making training time unfeasibly long. Support Vector Machines are particularly well suited to this case.

### 3.6 特殊案例(Special cases)
Some learning algorithms make particular assumptions about the structure of the data or the desired results. If you can find one that fits your needs, it can give you more useful results, more accurate predictions, or faster training times.


## 4. 其他一些关键技术：

- 梯度下降
- 正则化
- 特征学习
- 降维(Dimensionality Reduction)
- Boosting
- Kernels
- 交叉验证
- Ensemble methods: http://scikit-learn.org/stable/modules/ensemble.html
- Lasso

## 5. 常见的理论知识点：

### 5.1 概率论与数理统计
- 常见的分布以及相关性质
- 极大似然估计

### 5.2 线性代数
- 矩阵的半正定性
- 对偶
- 线性投影
- 奇异值分解
- 特征值与特征向量

### 5.3 优化理论
- 过拟合与欠拟合(bias-variance)
- 凸优化

### 5.4 其他
- 高维空间中距离的度量(范数)
- 算法的复杂度
- 自动编码
- VC维: 可以用来估计泛化错误的大小
- 最小二乘法

## 6. 常见机器学习任务汇总：

- 分类(区分不同疾病的亚型、对电影或书籍的评论进行分类、识别垃圾邮件)
- 手写体数字的识别(ORC)
- 推荐系统
- 异常检测(防信用卡欺诈)
- 图像识别(物体识别、医学图像识别)
- 自动驾驶
- 语音识别
- 聚类分析

## 7. Reference

- RoadToDataScientist.png: http://nirvacana.com/thoughts/2013/07/08/becoming-a-data-scientist/
- How to choose algorithms for Microsoft Azure Machine Learning
  link: https://docs.microsoft.com/en-us/azure/machine-learning/studio/algorithm-choice
- Machine learning algorithm cheat sheet for Microsoft Azure Machine Learning Studio
  link: https://docs.microsoft.com/en-us/azure/machine-learning/studio/algorithm-cheat-sheet
- Machine learning basics with algorithm examples
  link: https://docs.microsoft.com/en-us/azure/machine-learning/studio/basics-infographic-with-algorithm-examples
- scikit-learn_ml_map.png: http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
- Wu X, Kumar V, Quinlan J R, et al. Top 10 algorithms in data mining[J]. Knowledge and information systems, 2008, 14(1): 1-37.
- https://www.dezyre.com/article/top-10-machine-learning-algorithms/202
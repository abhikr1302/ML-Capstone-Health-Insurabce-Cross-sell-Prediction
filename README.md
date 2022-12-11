# ML-Capstone-Health-Insurance-Cross-sell-Prediction

# Introduction:-
Insurance is a means of protection from financial loss in which, in exchange for a fee, a party agrees to guarantee another party compensation in the event of a certain loss, damage, or injury. The amount of money charged by the insurer to the policyholder for the coverage set forth in the insurance policy is called the Premium.

Vehicle insurance (also known as car insurance, motor insurance, or auto insurance) is insurance for cars, trucks, motorcycles, and other road vehicles. Its primary use is to provide financial protection against physical damage or bodily injury resulting from traffic collisions and against liability that could also arise from incidents in a vehicle.

# Problem Statement:-
Our client is an Insurance company that has provided Health Insurance to its customers. Now they need your help in building a model to predict whether the policyholders (customers) from past year will also be interested in Vehicle Insurance provided by the company.
Building a model to predict whether a customer would be interested in Vehicle Insurance is extremely helpful for the company because it can then accordingly plan its communication strategy to reach out to those customers and optimize its business model and revenue.

# DATASET:-
Now, in order to predict, whether the customer would be interested in Vehicle insurance, we have information about following parameters

1. id : Unique ID for the customer
2. Gender : Gender of the customer
3. Age : Age of the customer
4. Driving License 0 : Customer does not have DL, 1 : Customer already has DL
5. Region Code : Unique code for the region of the customer
6. Previously Insured : 1 : Customer already has Vehicle Insurance, 0 : Customer doesn't have Vehicle Insurance
7. Vehicle Age : Age of the Vehicle
8. Vehicle Damage :Yes : Customer got his/her vehicle damaged in the past. No : Customer didn't get his/her vehicle damaged in the past.
9. Annual Premium : The amount customer needs to pay as premium in the year
10. Policy Sales Channel : Anonymized Code for the channel of outreaching to the customer i.e. Different Agents, Over Mail, Over Phone, In Person, etc.
11. Vintage : Number of Days, Customer has been associated with the company
12. Response : 1 : Customer is interested, 0 : Customer is not interested

Our data set contains 381109 rows and 12 columns. We have 5 numeric and 6 categorical independent features.
 
# Workflow:-
The workflow of this problem is divided into following sections-

I. Data Study - First, we will import python libraries such as NumPy and Pandas required to perform the study and then load the CSV file. Later we will perform some operations to understand and analyze the data.
 
II. Data Manipulation - Since Dataset contains int64, float64, object type of data and no Null Values in the Dataset, hence cleaning is not needed. We have handled outliers using the IQR method.

III. Correlation Graph

![download (24)](https://user-images.githubusercontent.com/110467640/206907509-5eac7564-ee0a-4a80-bfba-96dee69494a6.png)

We can see that all the features are very less correlated to each other.

IV. EDA and Visualisation:-

1. Response of each Gender

![download (25)](https://user-images.githubusercontent.com/110467640/206907574-204c69f6-a260-47c9-a0c1-2543ab4aa9b9.png)

This graph shows that Males prefer to take Vehicle Insurance as compared to Females.

2. Response of every Age

![download (26)](https://user-images.githubusercontent.com/110467640/206907698-766299de-deb6-4d9e-a23f-0bde97fda078.png)
![download (27)](https://user-images.githubusercontent.com/110467640/206907704-71b57d5c-63ef-472d-b471-84de5c1529ce.png)

We have divided the age of people into 3 groups into 'Young Age', 'Middle Age' and 'Old Age' based on their age criteria. This graph shows that most of the people are from Young Age, hence comprising 56.8% of total population.

3. Response of person having driving license

![download (28)](https://user-images.githubusercontent.com/110467640/206907773-373df76c-18ec-4bf7-b161-623a3ced84b2.png)

This graph shows that people having Driving License are more expected to take Vehicle Insurance.

4. Response in different Regions

![download (29)](https://user-images.githubusercontent.com/110467640/206907806-fa660ac5-543a-4056-ad40-f00b80794bdf.png)

It shows that Region code-38 has the highest number of possibilities for sales of Vehicle Insurance.

5. Response of persons who have previously insured

![download (30)](https://user-images.githubusercontent.com/110467640/206907859-3287ac4e-9df8-4e2d-a3c9-8f650651b355.png)

It clearly says that people who are not previously insured are more likely to take Vehicle Insurance.

6. Analysis on Vehicle age

![download (31)](https://user-images.githubusercontent.com/110467640/206907950-e1cab28c-75c2-4c81-929b-481f0b9463ec.png)

This graph says that people who have a vehicle for 1-2 years are more interested in taking Vehicle insurance.

7. Effect of Vehicle damage

![download (32)](https://user-images.githubusercontent.com/110467640/206907992-0cacd366-ddde-4e07-91a8-2eb8b23afe45.png)

Person facing Vehicle damage is more keen towards taking vehicle insurance as depicted in the above graph.

8. Distribution of Annual premium

![download (33)](https://user-images.githubusercontent.com/110467640/206908029-c3d2c63e-788a-4a6a-b4b6-94296e4eb357.png)
![download (34)](https://user-images.githubusercontent.com/110467640/206908035-9f60df82-9e13-4b6e-8512-3876a980ec3c.png)

We have grouped different annual premiums into 5 groups namely- 'Very Low Premium', 'Low Premium', 'Medium Premium', 'High Premium', 'Very High Premium'.
We can see that most of the Annual premium lies in High premium(between 30K and 40K).
Most of the people are interested to pay an annual premium in the range of 30K-40K.

9. Response towards Policy Sales Channel

![download (35)](https://user-images.githubusercontent.com/110467640/206908085-0edc9fbe-502c-4e24-9e86-0ada2232c7ac.png)
![download (36)](https://user-images.githubusercontent.com/110467640/206908090-5802f70c-4f8b-4ac1-8a65-378b33f61d17.png)

It shows that Policy sales channel 150-160 has outreached the customer more as compared to others.
We have grouped Policy sales channel into 4 different groups i.e. 'Channel A', 'Channel B', 'Channel C', 'Channel D'.
This states that maximum of Policy sales has been booked by Channel D.

10. Vintage of Customers

![download (37)](https://user-images.githubusercontent.com/110467640/206908135-fc0d6e31-ea8b-4d60-a980-2034389588b6.png)
![download (38)](https://user-images.githubusercontent.com/110467640/206908140-b5a943a4-eeb7-4354-b4fd-b0fc8854a497.png)

The distribution of Vintage Column is evenly spread which ranges from 10 days to 300 days.
We have grouped Vintage columns into 3 types of customers- 'New Customer', 'Intermediate Customer', 'Old Customer').
This shows that all three types of customers are nearly equal in number.

V. Data Processing

1. Label Encoding- Label Encoding refers to converting the labels into a numeric form so as to convert them into the machine-readable form.

![Screenshot 2022-12-01 210243](https://user-images.githubusercontent.com/110467640/206908257-dab3b206-ab93-4e6a-ac79-09d2b88927d2.png)

We have encoded categorical features like ‘Gender’, ‘Age Group’, ‘Vehicle Age’, ‘Vehicle Damage’, ‘Policy Sales Channel Group’, ‘Vintage Group’, ’Annual Premium Group’ into defined codes using Label Encoding.

2. Feature Selection- We have done feature selection using Mutual Information. Mutual information is calculated between two variables and measures the reduction in uncertainty for one variable given a known value of the other variable.

![image](https://user-images.githubusercontent.com/110467640/206908296-6faa8f9f-4663-4ed1-b706-b7c37cb5e234.png)

From the above bar plot, we can conclude that Previously Insured is the most important feature and has the highest impact on dependent features and Region Code is the least dominant feature.

3. Handling Imbalanced Dataset- Since our dataset was highly imbalanced, we have used the ‘SMOTE’ process to treat it. SMOTE is an oversampling technique where the synthetic samples are generated for the minority class to solve the imbalance problem.

![download (39)](https://user-images.githubusercontent.com/110467640/206908421-8d530af0-3a7f-4742-a578-5dfe5d107ce9.png)
![download (40)](https://user-images.githubusercontent.com/110467640/206908428-49841558-238e-4d4f-940c-c0a488d716a6.png)
       
4. Train-Test Split & Hyperparameter Tuning- We have splitted our dataset into Train and Test data in the ratio of 80:20. We have used Halving Random Search cv and Random Search cv for hyperparameter Tuning.

![64801HPTT](https://user-images.githubusercontent.com/110467640/206909220-7c214561-cbfa-4c43-91d3-c77b884460a3.png)

5. One Hot Encoding- One hot encoding is a process by which categorical variables are converted into a form that could be provided to ML algorithms to do a better job in prediction.

![Screenshot 2022-12-01 224147](https://user-images.githubusercontent.com/110467640/206909374-4d8a1a4b-22ed-42f7-8223-18e41a156e46.png)

VI. Model Building

1. Decision Tree Classifier- Decision tree is a flowchart-like tree structure, where each internal node denotes a test on an attribute, each branch represents an outcome of the test, and each leaf node (terminal node) holds a class label. 
     
2. Random Forest Classifier- The random forest algorithm is made up of a collection of decision trees, and each tree in the ensemble consists of a data sample drawn from a training set with replacement, called the bootstrap sample.

3. XgBoost Classifier- XGBoost, which stands for Extreme Gradient Boosting,  is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework and provides parallel tree boosting.

4. Naive bayes Classifier- A Naive Bayes classifier is a probabilistic machine learning model  based on Bayes' Theorem that is used for classification task.

5. K Nearest neighbors(KNN)- The k-nearest neighbors algorithm, also known as KNN or k-NN, is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point.

6. Light Gradient Boosting(LGBM)- LightGBM is a gradient boosting framework based on decision trees to increases the efficiency of the model and reduces memory usage. 

VI. Comparison Metrics

![Screenshot (2)](https://user-images.githubusercontent.com/110467640/206910001-35babe52-3671-4a20-868e-9c50ba93145b.png)

After comparing each model, we found out that Random Forest is having the maximum Accuracy among all models, hence it is the best model to train.

VII. Conclusion:-
1. Males prefer to take Vehicle Insurance as compared to Females.
2. Middle Age Population(30-50 yrs) are more likely to take Vehicle Insurance.
3. Majority of people are interested in paying an annual premium in the range of 30k-40k(High Premium).
4. Sales Channels 140-160 is the most contacted channel among all the channels and it has provided highest no. of conversion. 

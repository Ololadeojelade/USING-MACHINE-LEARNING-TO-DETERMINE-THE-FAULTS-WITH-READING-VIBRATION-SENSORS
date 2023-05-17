# USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS

For this task, a manufacturing company wants to know if there are faults with the readings derived from their vibration sensors. The results derived after this task would help them decide on the predictive maintenance. 

## DATA PREPARATION AND IMPLEMENTATION

After logging into the Databricks workspace, the downloaded fault dataset is imported into the workspace using the browse file option. A new cluster is created and computed. For this course work the machine learning task cluster was created. After creating the cluster, a new notebook was created. This notebook is used to input the runnable cells.

To adequately execute this task, it is important to implement the machine learning algorithm was implemented. This function mlflow.pyspark.ml.autolog() makes it possible for hyperparameters, metrics, and model faults to be automatically logged without any additional changes.

![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/6e157918-8393-4018-8933-2781614bf977)


![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/d7ff3079-5a7f-4afb-9194-03109cddf2b5)


A new data frame FaultdatasetDF was created using spark.read.csv to enable the FaultDataset file to be read as FaultdatasetDF. The header was made true because from the faultdataset file we have a header present there. The inferSchema was also set to be true.

![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/9325e691-4db2-43b0-ab99-09166a52d5be)


![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/6e89a7d4-e488-4614-8b39-d4e6ba9cc8a9)


![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/d5f291e7-f263-4c41-adba-0757e7e50abb)


![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/0e229003-15db-4ae2-8812-e5dc50289336)


The images above show the values present in the fault dataset. The data profile to generate a summary of the dataset including count, mean, standard deviation, minimum and maximum values of each numerical feature, while the visualization gives a pictorial view of the fault data set profile.

## PREPROCESSING THE DATA

![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/c19fd389-c7c1-4b12-a9f9-a53e300e7141)


From pyspark.ml.feature the RFormula was imported. The RFormula is important for pre-processing data because it is a formula used for supervised learning problem where the objective is to predict a target variable based on one or more input features. It also can handle a variety of data formats, including categorical and continuous variables in machine learning.

The existing features column was dropped from the Faultdataset column using drop().
To show what we want to predict in the FaultdatasetDF data frame "fault_detected ~ ."  was used as the target variable.
The FaultdatasetDF has the RFormula transformation applied to it by the preprocess.fit(FaultDatasetDF).transform(FaultDatasetDF) function. The fit() function was implemented to determine the transformation based on the inputted DataFrame, while the transform() method applies the transformation to create the new DataFrame with the target variable and input features.

![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/104521ca-5a4d-4145-9b67-ad3e9442202a)

![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/cb1e2211-2ee9-4fe6-b0d5-ed1110377753)


From the results derived, a new column label has been added to the table and it contains the same values as the fault_detected column.

![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/bbfefde8-701e-47c0-abd1-9f35d67b5821)


For the Faultdataset, we would be splitting the data set in 70 and 30. 70% of the data set would be used as the training data set while 30% of the data would be used as test data set. This is because when executing supervised learning, we need to set aside some of the dataset when a model is being trained. The data set that is being set aside can be used to make predictions in a new data and we can therefore see how the data performs.

![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/84c467b9-dbe5-45cb-ab66-064c59aa9967)


The pyspark.ml.classification was used to import the decision tree classifier. A decision tree classifier is a supervised learning algorithm that is used for classification and regression modelling. Regression is a method used for predictive modelling, so these trees are used to either classify data or predict data. 
A DecisionTreeClassifier object was created to specify the labelCol variable as the name of the label column in the training dataset while the featuresCol variable was used as the name of the column identifying the training features.
The fit function was used to fit the DecisionTreeClassifier to the training data trainingDF. The model is then used to make necessary predictions on the trained data.

## EVALUATING THE MODEL

![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/97139e57-1e67-4476-98fb-f1beb4102e2e)


![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/a1e3c47f-3b9a-4c1d-b9dd-ac33db6ed0b5)


![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/eb4f0d21-568c-4885-ad48-15f25b1c139e)


![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/59e52ca5-d40f-4914-91f5-300415444467)


![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/c4c8d6bb-f416-4a21-95ad-6d0b9d8fb0c3)


For the model to be evaluated, the trained model would be used in making predictions on the test data that was set aside. On the test data, we applied the transform() technique to produce this predictions.

From the result derived, there was an addition of the raw prediction column, probability column and the prediction column along sides the existing columns in the file. The probability column is a vector containing the predicted probability that the example belongs to each vibration sensors. The prediction column predicts which vibration sensor has the highest predicted probability.

An evaluator needs to be created to determine the effectiveness of the model. For this coursework, accuracy would be used as the evaluation model.

![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/eb666f18-663c-4381-98fa-be730b935509)


The Multiclass Classification Evaluator was implemented to determine the accuracy of this model. The Multiclass Classification Evaluator classifies each observation in a dataset into one of many categories.
With the result derived, the accuracy is 0.952432, which means that 95% of the predictions made by our model on the test dataset are accurate. This simply means that this model can be a reliable model for prediction. 
The result of the experiment logged in the MLflow where decision tree classifier was fit shows the 95% accuracy of the model.

![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/11f24716-a64f-433b-b8f7-0bd8efa8b0c2)


These hyperparameters are parameters that influence the training process of the dataset.
The impurity gini here measures the number of times the attributes can be split on each branch node.
The maxBins 32 measures the different ways the algorithm can split the data on a specific attribute.
The maxDepth 5 measures that number of times the model is allowed to split before the leaf node is terminated.


## THE USE OF PARAGRIDBUILDER AND TRAIN VALIDATION SPLIT FOR GRID SEARCH.

![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/3af69e65-3ac4-4593-a0a8-f0796bed71cf)


The pyspark.ml.tuning was used to import ParaGridBuilder into the workspace.
The ParaGridBuilder is assigned to the parameter variable. The impurity hyperparameter could use either gini or entropy for the decision tree splits. The maxDepth hyperparameter could use either 3, 5, or 7 maximum depths for the decision tree. The maxBins hyperparameter could use 16,32, or 64 as the maximum number of bins used for separating characteristics of continuous data.

![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/1ce7793e-1d61-4968-aec8-9b79fc8b043a)


To execute that train validation split, pyspark.ml.tuning was used to import the TrainValidationSplit. Tvs was assigned to the TrainValidationSplit. 70% of the dataset was assigned to be used for the training model out of the entire 100%.
The .setEstimatorParamMaps(parameters)sets the hyperparameter grid created earlier as the parameter grid to be used.
The .setEstimator(dt) sets the decision tree estimator to be used for training the data set.
The .setEvaluator(evaluator) sets the evaluation metric to be used for selecting the best hyperparameters during the tuning process.

![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/c3702a6f-980e-4433-ad6d-0676063f8b64)


The fit() method was used to perform the hyperparameter tuning. After the grid search, the best performing model can now be performed.

![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/2c6b84de-22f9-4d48-a9c8-7a68e6652b0a)


From the result derived, the best maxDepth parameter to be used for this model is 7, the best impurity parameter is gini and the best maxBins parameter is 64.

![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/344011d6-4112-4d34-8b74-b210dfe09173)


After using the best model of the test data set to make predictions. The result derived from the test data set shows a 95% accuracy.

![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/28a4d7dc-ad9a-4bda-ae8b-5e3c671b526d)


![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/e086b4c5-7637-489e-91c6-c523a7be34ba)


To make predictions, the model is loaded, and the logged model is used.

![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/a045a871-47df-414f-8d5c-fc5e11b0af7e)


The test dataset contains the predictions produced by the loaded model. This can be helpful for determining how effectively the trained model generalises beyond the training data and how well it performs on new data.

![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/68f17329-e2f4-4c63-acd7-4c151686588c)


The result shows the details of predictions on the loaded model.

Asides from the decision tree classifier, the random forest classifier and the logistics regression classification methods were used. Logistics regression classification methods is a supervised learning that is used to predict observations to a discrete set of classes while the random forest classifier is combines prediction from other models.

![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/0bd06ea4-e3c0-48cc-8125-52653f4383a2)


![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/cb8d9c55-3690-4d26-ad9e-2d3dfa742759)


![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/3de09115-409e-4656-8314-cf6184959ad9)


The accuracy derived after using the random forest classifier type of classification on this model is 96%. This means that this classification method can be used on this model.

![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/ec0e1cff-ac31-43e0-8bc3-ee9f616337ed)


![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/e010b14c-f128-4c56-88c3-3212b2a584ab)


![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/7dd51dae-0f15-4684-aa77-5c0ac740caf3)


From the result derived, the best maxDepth parameter to be used for this model is 7, the best impurity parameter is gini and the best maxBins parameter is 10.

![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/d2cc60a8-9c36-4c78-a585-ab7f871aa011)


![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/d97bb62d-f359-4d3d-9246-1b094ff16c5e)


![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/7b309dc2-37b0-4a5a-91ac-bfb3ff7bd1af)


The accuracy derived after using the logistics regression classifier type of classification on this model is 80%. This means that this classification method good but not as good as the decision tree classification and the random forest classification method.

![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/aa5481b9-1032-4fbb-ab8b-ba222f5b3afe)


![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/48a75691-ebc8-438e-bcac-6b7728088a8c)


![image](https://github.com/Orlawlardey/USING-MACHINE-LEARNING-TO-DETERMINE-THE-FAULTS-WITH-READING-VIBRATION-SENSORS/assets/124607057/1b0824f1-5fd0-42c6-961a-2dda0fc273ff)


From the result derived, the best regParam to be used for this model is 0.01, the best fitIntercept parameter is True and the elasticNetParam parameter is 0.

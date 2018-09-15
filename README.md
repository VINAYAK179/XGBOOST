# XGBOOST
XGBOOST Classifier

Conceptual Approach : 
I have neglected the first column which is time. As the data contains some missing values, I have replaced them with the mean value in both training and testing data. To increase the performance of the model, I have shuffled the data. To measure the accuracy division of training data is made in the ratio of 80% and 20% repectively. As the labels in testing data are missing, the recent split is used for measuring accuracy. Then again the classifier is again trained on training data and tested on the given test data . 

Model Performance : 
The model performance when the given training data is divided and used is 99.15 % . This accuracy is calculated using the confusion matrix . 

The complexity is not high . The model takes approx time of 5 minutes to run . 

If I had more time I would like to use Deep Learning . Long Short Term Memory(LSTM) can be used for better performance .  

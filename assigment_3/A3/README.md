### Text Classification using RNN and LSTM

1. Download the libraries
```
pip3 install tensorflow numpy
```
```
pip3 install keras
```

## Dataset
We used a dataset called 20 Newsgroups which ontained 20,000 newsgroup documents that were labeled from one of 20 categories. The data was split into two folders; one for the validation set and one for the test. 

## How to run the code
```
python3 assignment3.py
```
To run the specific models, uncomment the code within the main function
We also provided a graphs.py file that contains code on how we were able to create graphs to show the progression of the validation accuracy while training the model. 
```
python3 graphs.py
```

Running these commands will 
* load the dataset
* preprocess the data
* build the models
* train the models on the train set
* test the performance on the val and test set

## Information on the models
1. Recurrent Neural Network (RNN): A Bidirectional SimpleRNN model, which uses recurrent connections to handle sequence data.
2. Bidirectional LSTM (Long Short-Term Memory): A Bidirectional LSTM model that learns both forward and backward dependencies in the text data.


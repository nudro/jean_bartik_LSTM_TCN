# jean_bartik_LSTM_TCN
Google Colaboratory-runnable notebooks for time series forecasting on the PAMAP dataset

Notebooks designed to run on Google Colaboratory from the Jean Bartik Computing Symposium 2019 in West Point Miitary Academy.

Dataset is Subject 102: Download from the .zip file (`.dat`) file here: https://archive.ics.uci.edu/ml/machine-learning-databases/00231/

<b>Purpose of the “Build a RNN” Workshop</b>
This workshop is designed for students who are just learning Python and seeking more examples of intermediate code, to students who have dabbled in machine learning and are curious about deep learning models.  No pre-requisites are required, just an interest to learn! We hope that you’ll be able to learn how to use a Jupyter Notebook, learn how to process data using Python, and also get some exposure to two important algorithms when dealing with time, sequence-based data. 

<b>Introduction to Recurrent Neural Networks (RNN)</b>

For the RNN and the WaveNet sessions, our main dataset will be the PAMAP2 Physical Activity Monitoring Data Set, focusing on one test subject’s sensor data. In the RNN tutorial you’ll be learning how to build both a “vanilla” RNN using only the Python numpy library and a Long Short-Term Memory (LSTM) RNN in Keras. We will build out an RNN in numpy (see: rnn_numpy.ipynb) to demonstrate the fundamentals of deep learning, tensors (otherwise called matrices) and operations on them. For the numpy RNN demonstration we will shift to using a song lyrics datatset to show how an RNN can be trained for character-by-character sequence generation. 

With that fundamental understanding of basic RNN code, we will embark on building out an LSTM for time series prediction on the PAMAP2 dataset using the Keras library ((see: tutorial_rnn_lstm.ipynb). With the ability to store memory in an “internal state,” these networks are quite effective in predicting sequences, such as time series and text. 

You need to first begin by opening the (see: tutorial_rnn_lstm.ipynb) file. We will then open the rnn_numpy.ipynb file.

<b>Introduction to 1D Convolutional Nets Using WaveNet</b>

Recent research suggests that using convolutional neural networks (CNNs) are just as effective at sequence prediction tasks and have advantages in that they take less time to train and are more interpretable. For a “bake off” we will use the same PAMAP2 sensor data set but apply it on a one-dimensional (non-spatial) CNN using the basic version of a popular temporal convolutional network called “Wavenet” developed by Google as the primary machine language translation algorithm. Wavenet in its basic form can be implemented in Keras which is what we will do today (see: tutorial_wavnet.ipynb). Let’s take a look at how Wavenet using a temporarl CNN works in comparison to the LSTM. We will also introduce new ways to process the sensor data to prepare it for the CNN. 


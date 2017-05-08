# Predicting-Crime-in-Chicago
Capstone Project with General Assembly's Data Science Immersive Course

## Presentation

If you would like to see my presentation in full then please follow this link (https://www.youtube.com/watch?v=CkJ9vXZzvGs&t=14s) and you can follow along with the slides that I have published here (https://docs.google.com/presentation/d/15WWtUiy4eWNHLzmTgsFPvc30j7INQzuPfNzq8Nd2mPo/pub?start=false&loop=true&delayms=3000)

## Data

The Chicago Data Portal (https://data.cityofchicago.org/browse?&page=1) logs all reported crimes and abandoned lots since 2010 and DarkSky API (https://darksky.net/dev/) has logged weather records that date back decades.

All data was indexed by day and lagged by one day so as to make my project as close to the real world as possible. All data was also segmented by Ward and the steps are shown in the notebook labeled Technical Report.ipynb within the Capstone folder.

With the help of Sci-kitLearn, I assembled pipelines to organize and transform incoming data (instructions can also be found in the above mentioned Jupyter notebook).

## Modeling Techniques

The reader can follow my step by step walkthrough of a SARIMA times series model in the Jupyter Notebook labeled Technical Report. The other modeling techniques explored here include a neural network (MLPClassifier) baggging and boosting ensemble methods (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier and ExtraTreesClassifier), KNeighborsClassifier, LogisticRegression and MulitnomialNB which are all part of the Sci-kitLearn package of programs.

When running neural network and voting classifier experiments, I operationalized y into two classes. A 1 if the nunmber of batteries on a given day fell in the upper 25th percentile of batteries for that particular ward and a 0 otherwise. 

## Results

The highest f-1 score reported here was 0.51 for the days labeled 1 (dangerous days) and it was accomplished by running a neural network on Ward 17 with 1 hidden layer consisting of 7 neurons. The script used to arrive at that number can be found under the python file labeled nn_17.py inside of the Capstone folder.

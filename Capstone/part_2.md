Predicting Chicago Crime 

Problem Statement
To accurately predict the number of violent crimes (specifically battery) that will occur in beats across Chicago. 

Approach
At the mininum, the features that I plan to include in my model are the number of thefts, batteries and assaults across Chicago each day, weather (including temperature, humidity, barometric pressure and rainfall) and the number of libraries, schools and police stations in each beat. 

Stretch features that I would like to include are NLP statistics on tweets across Chicago, NLP statistics on recent news (LDA topics could include police brutality and local protests/rallies) and the times and locations of reported grafiti. 

Shortcomings of Data
I am depending on the accuracy of Chicago's public data portal which can lead to problems. For example, some entries can be duplicate entries of the same crime (one entry for each criminal or multiple cops entering information on the same crime). Also, the date that the information was entered into Chicago's system does not always match up with the date that the crime was committed which again, leads us to inaccuracies. The City of Chicago also witholds inforamtion on crimes in which a murder occurred. I have submitted a request via the help of the Freedom of Information Act but the wait time is vague (i.e. I might receive this data when it is too late to model with).  

An inherent risk is that I find no correlation between dangerous behavior and crimes. It would not be ridiculous to say that criminal behavior is brought on by random acts of chance that cannot be predicted (spontaneous disagreements that grow out of control or marital disputes that are brought on by any number of variables like losing one's job or child custody suits, etc.). 

Success
I would consider it a success if I can produce a model that predicts a realistic number of batteries in each beat. As it now stands, my predictions are both ambiguous and inaccurate.

Stretch success would be to discover a feature that significantly reduces the likelihood of crime to occur. For example, I would like to find the correlation between a beat's count of libraries, schools and hospitals and its battery count. My hope is that more of the former leads to less of the latter. 

EDA 
I have found that crime in Chicago has steadily decreased year over year for the most part and that we consistently see upticks in crime during the summer (seasonality). Another timeseries trend that makes itself aparent is that more crimes occur in the latter half of the week than in the beginning. Also, the majority of crimes occur on the South and West sides.

A problem that I have discovered while exploring the data is a lack of consistency within beats. As can be seen in the graph (labeled plotBeat(2533)) spikes in crime occur sporadically in months that we wouldn't expect crime to spike. The same holds true for plotBeat(815) and plotBeat(2223). Such spikes make timeseries predictions very dificult. 
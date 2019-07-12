'''The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, 
the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the 
international community and led to better safety regulations for ships. One of the reasons that the shipwreck led to such loss of life 
was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the
sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class. In this challenge, 
we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of 
machine learning to predict which passengers survived the tragedy. 
'''

#Since code is written in Jupyter Notebook, print statements will be missing at some places

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split

train = pd.read_csv('../../../../train.csv')
test = pd.read_csv('../../../../test.csv')
train.head()

#Chcking if any Null Values exist
print(train.isnull().sum()(

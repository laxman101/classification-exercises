#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# import my acquire module
import acquire


# ### Iris Data

# In[2]:


def clean_iris(iris_df):
    '''
    This function will clean the data...
    '''
    iris_df = iris_df.drop(columns='species_id')
    iris_df.rename(columns={'species_name':'species'}, inplace=True)
    dummy_df = pd.get_dummies(iris_df[['species']], dummy_na=False)
    iris_df = pd.concat([iris_df, dummy_df], axis=1)
    
    return iris_df


# In[3]:


def split_iris_data(df):
    '''
    Takes in a dataframe and return train, validate, test subset dataframes
    '''
    train, test = train_test_split(iris_df, test_size = .2, random_state=123, stratify=iris_df.species)
    train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train.species)
    
    return train, validate, test


# ### Titanic Data

# In[4]:


def clean_titanic(df):    
    df.drop(columns=['passenger_id', 'class', 'embarked', 'deck'], axis=1, inplace=True)
    df = pd.get_dummies(df, columns=["sex", "embark_town"], drop_first=[True, True])
    return df   


# In[5]:


def split_data(titanic_df):
    '''
    Takes in a dataframe and return train, validate, test subset dataframes
    '''
    train, test = train_test_split(titanic_df, test_size = .2, random_state=123, stratify=titanic_df.survived)
    train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train.survived)
    
    return train, validate, test


# ### Telco Dataset

# In[6]:


def clean_telco(telco_df):
    '''
    This function will clean the telco data.
    '''
    telco_df = telco_df.drop_duplicates()
    cols_to_drop = ['payment_type_id', 'internet_service_type_id', 'contract_type_id']
    telco_df = telco_df.drop(columns=cols_to_drop)
    dummy_df = pd.get_dummies(telco_df[['gender', 'contract_type', 'internet_service_type', 'payment_type']], dummy_na=False, drop_first=[True, True, True, True])
    telco_df = pd.concat([telco_df, dummy_df], axis=1)
    
    return telco_df


# In[7]:


def split_data(telco_df):
    '''
    Takes in a dataframe and return train, validate, test subset dataframes
    '''
    train, test = train_test_split(telco_df, test_size = .2, random_state=17, stratify=telco_df.churn)
    train, validate = train_test_split(train, test_size=.3, random_state=17, stratify=train.churn)
    
    return train, validate, test


# In[8]:


def prep_telco(telco_df):
    '''
    This function will clean the telco data...
    '''
    #Drop Duplicates
    telco_df = telco_df.drop_duplicates()
    
    # Drop null values stored as whitespace    
    telco_df['total_charges'] = telco_df['total_charges'].str.strip()
    telco_df = telco_df[telco_df.total_charges != '']
    
    # Convert to correct datatype
    telco_df['total_charges'] = telco_df.total_charges.astype(float)
    
    # Drop Columns
    cols_to_drop = ['customer_id', 'payment_type_id', 'internet_service_type_id', 'contract_type_id']
    telco_df = telco_df.drop(columns=cols_to_drop)
    
    # Get dummies for non-binary categorical variables
    dummy_df = pd.get_dummies(telco_df[['multiple_lines',                               'online_security',                               'online_backup',                               'device_protection',                               'tech_support',                               'streaming_tv',                               'streaming_movies',                               'contract_type',                               'internet_service_type',                               'payment_type']], dummy_na=False)
    # Concatenate dummy dataframe to original 
    telco_df = pd.concat([telco_df, dummy_df], axis=1)
   
    return telco_df


# In[ ]:





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 12:03:58 2018

@author: bennicholl
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer

#test = '/Users/bennicholl/Desktop/credit_default_risk/application_test.csv'

train = '/Users/bennicholl/Desktop/credit_default_risk/application_train.csv'
train = pd.read_csv(train)

"""this function runs a pca on all the below features which are associated with housing situation"""
def housing_pca(train = train):
    ## I might turn nan values into something else, such as 0.01
    """turn nan values into 0"""
    train['COMMONAREA_AVG'] = train['COMMONAREA_AVG'].replace(np.nan, 0)
    train['BASEMENTAREA_AVG'] = train['BASEMENTAREA_AVG'].replace(np.nan, 0) 
    train['YEARS_BUILD_AVG'] = train['YEARS_BUILD_AVG'].replace(np.nan, 0)
    train['FLOORSMAX_AVG'] = train['FLOORSMAX_AVG'].replace(np.nan, 0)
    train['APARTMENTS_AVG'] = train['APARTMENTS_AVG'].replace(np.nan, 0)
    train['LANDAREA_AVG'] = train['LANDAREA_AVG'].replace(np.nan, 0)
    train['LIVINGAREA_AVG'] = train['LIVINGAREA_AVG'].replace(np.nan, 0)
    train['NONLIVINGAREA_AVG'] = train['NONLIVINGAREA_AVG'].replace(np.nan, 0)
    """run pca on the above panda columns"""
    pca = PCA(n_components=2)
    pca.fit_transform(X= [train['COMMONAREA_AVG'], train['BASEMENTAREA_AVG'], train['YEARS_BUILD_AVG'], train['FLOORSMAX_AVG'],
            train['APARTMENTS_AVG'], train['LANDAREA_AVG'], train['LIVINGAREA_AVG'], train['NONLIVINGAREA_AVG'] ])
    """drop all the columns we ran a pca on"""
    train = train.drop(['COMMONAREA_AVG', 'BASEMENTAREA_AVG', 'YEARS_BUILD_AVG', 'FLOORSMAX_AVG',
                        'APARTMENTS_AVG', 'LANDAREA_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAREA_AVG'], 1)
    """append the three principal components as colums to the dataframe"""
    train = train.assign(house_component_one = pca.components_[0])
    train = train.assign(house_component_two = pca.components_[1])
    return train

train = housing_pca()

"""runs a pca on features that are specifically assocaited with defaulting on various charges"""
def default_pca(train = train):
    pca = PCA(n_components=1)
    
    train['DEF_60_CNT_SOCIAL_CIRCLE'] = train['DEF_60_CNT_SOCIAL_CIRCLE'].replace(np.nan, 0)
    train['OBS_60_CNT_SOCIAL_CIRCLE'] = train['OBS_60_CNT_SOCIAL_CIRCLE'].replace(np.nan, 0)
    train['DEF_30_CNT_SOCIAL_CIRCLE'] = train['DEF_30_CNT_SOCIAL_CIRCLE'].replace(np.nan, 0)
    train['OBS_30_CNT_SOCIAL_CIRCLE'] = train['OBS_30_CNT_SOCIAL_CIRCLE'].replace(np.nan, 0)
    
    pca.fit_transform(X = [train['DEF_60_CNT_SOCIAL_CIRCLE'], train['OBS_60_CNT_SOCIAL_CIRCLE'], 
                      train['DEF_30_CNT_SOCIAL_CIRCLE'], train['OBS_30_CNT_SOCIAL_CIRCLE']])
    train = train.drop(['DEF_60_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 
                       'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_30_CNT_SOCIAL_CIRCLE'], 1)
    train = train.assign(defaults_component = pca.components_[0])
    return train
    
train = default_pca()
    

"""this drops all of the flag docs except doc 3, 6, and 8"""
def drop_flag_docs(train = train):
    train = train.drop(['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_7',
                       'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 
                       'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19',
                       'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21'], 1)
    
    return train

train = drop_flag_docs()

"""for now I'm dropping credit inquiries, but i may run a pca on them"""
def drop_credit_inquiries(train = train):
    train = train.drop(['AMT_REQ_CREDIT_BUREAU_YEAR', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_MON',
                        'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_HOUR'], 1)
    return train

train = drop_credit_inquiries()


"""drops the NAME_TYPE_SUITE column, which has to do with who the individual came with when he applied for loan"""
def drop_came_with(train = train):
    train = train.drop(['NAME_TYPE_SUITE'], 1)
    
    return train

"""drops the days_employed column because the data is dicked up"""
def drop_days_employed(train = train):
    train = train.drop(['DAYS_EMPLOYED'], 1)
    
    return train

train = drop_came_with()

"""turns our NAME_EDUCATION_TYPE column into binary values pertaining to their education level"""
def education_type(train = train):
    """create binary academic degree column"""
    train['Academic degree'] = train['NAME_EDUCATION_TYPE'].replace('Academic degree', 1)
    train['Academic degree'] = train['Academic degree'].replace(['Higher education', 'Incomplete higher','Lower secondary','Secondary / secondary special'] , 0)
    """create binary Higher educationcolumn"""
    train['Higher education'] = train['NAME_EDUCATION_TYPE'].replace('Higher education', 1)
    train['Higher education'] = train['Higher education'].replace(['Academic degree', 'Incomplete higher','Lower secondary','Secondary / secondary special'] , 0)
    
    """create binary Incomplete higher column"""
    train['Incomplete higher'] = train['NAME_EDUCATION_TYPE'].replace('Incomplete higher', 1)
    train['Incomplete higher'] = train['Incomplete higher'].replace(['Academic degree', 'Higher education','Lower secondary','Secondary / secondary special'] , 0)
    """create binary Lower secondary column"""
    train['Lower secondary'] = train['NAME_EDUCATION_TYPE'].replace('Lower secondary', 1)
    train['Lower secondary'] = train['Lower secondary'].replace(['Academic degree', 'Higher education','Incomplete higher','Secondary / secondary special'] , 0)
    
    """create binary Secondary / secondary special column"""
    train['Secondary / secondary special'] = train['NAME_EDUCATION_TYPE'].replace('Secondary / secondary special', 1)
    train['Secondary / secondary special'] = train['Secondary / secondary special'].replace(['Academic degree', 'Higher education','Incomplete higher','Lower secondary'] , 0)
    return train

train = education_type()


"""turns our NAME_FAMILY_STATUS column into binary values pertaining to their relationship status. doesn't include "unknown" status"""
def family_status(train = train):
    """create binary civil marriage column"""
    train['Civil marriage'] = train['NAME_FAMILY_STATUS'].replace('Civil marriage', 1)
    train['Civil marriage'] = train['Civil marriage'].replace(['Married', 'Separated','Single / not married','unknown', 'Widow'] , 0)
    
    """create binary married column"""
    train['Married'] = train['NAME_FAMILY_STATUS'].replace('Married', 1)
    train['Married'] = train['Married'].replace(['Civil marriage', 'Separated','Single / not married','unknown', 'Widow'] , 0)
    
    """create binary seperated column"""
    train['Seperated'] = train['NAME_FAMILY_STATUS'].replace('Separated', 1)
    train['Seperated'] = train['Seperated'].replace(['Civil marriage', 'Married','Single / not married','unknown', 'Widow'] , 0)
    
    """create binary civil marriage column"""
    train['Widow'] = train['NAME_FAMILY_STATUS'].replace('Widow', 1)
    train['Widow'] = train['Widow'].replace(['Civil marriage', 'Separated','Single / not married','unknown', 'Married'] , 0)   
    
    train = train.drop(['NAME_FAMILY_STATUS'],1)
    
    return train

train = family_status()

"""turns our NAME_HOUSING_TYPE column into binary values pertaining to their housinging situation"""
def housing_type(train = train):
    """create binary Co-op apartment column"""
    train['Co-op apartment'] = train['NAME_HOUSING_TYPE'].replace('Co-op apartment', 1)
    train['Co-op apartment'] = train['Co-op apartment'].replace(['House / apartment', 'Municipal apartment','Office apartment','Rented apartment', 'With parents'] , 0)
    
    """create binary House / apartmen column"""
    train['House / apartment'] = train['NAME_HOUSING_TYPE'].replace('House / apartment', 1)
    train['House / apartment'] = train['House / apartment'].replace(['Co-op apartment', 'Municipal apartment','Office apartment','Rented apartment', 'With parents'] , 0)

    """create binary Municipal apartment column"""
    train['Municipal apartment'] = train['NAME_HOUSING_TYPE'].replace('Municipal apartment', 1)
    train['Municipal apartment'] = train['Municipal apartment'].replace(['House / apartment', 'Co-op apartment','Office apartment','Rented apartment', 'With parents'] , 0)

    """create binary Office apartment column"""
    train['Office apartment'] = train['NAME_HOUSING_TYPE'].replace('Office apartment', 1)
    train['Office apartment'] = train['Office apartment'].replace(['House / apartment', 'Co-op apartment','Municipal apartment','Rented apartment', 'With parents'] , 0)

    """create binary Rented apartment column"""
    train['Rented apartment'] = train['NAME_HOUSING_TYPE'].replace('Rented apartment', 1)
    train['Rented apartment'] = train['Rented apartment'].replace(['House / apartment', 'Co-op apartment','Municipal apartment','Office apartment', 'With parents'] , 0)

    """create binary With parents column"""
    train['With parents'] = train['NAME_HOUSING_TYPE'].replace('With parents', 1)
    train['With parents'] = train['With parents'].replace(['House / apartment', 'Co-op apartment','Municipal apartment','Office apartment', 'Rented apartment'] , 0)
    
    train = train.drop(['NAME_HOUSING_TYPE'], 1)
    return train

train = housing_type()






# binary(-1, 1)
# contract_type, code_gender,

# binary(1, 0)
# flag_own_car, flag_own_reality, name_income_type

# divide by max
# cnt_children, 

# standard scaler
# AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE


"""I believe REGION_POPULATION_RELATIVE and DAYS_EMPLOYED are coorelated. runs a scatterplot"""
#REGION_POPULATION_RELATIVE
#DAYS_BIRTH
#DAYS_REGISTRATION   -   already scaled 
#DAYS_ID_PUBLISH
#OWN_CAR_AGE
#FLAG_MOBIL
#FLAG_EMP_PHONE
#FLAG_WORK_PHONE
#FLAG_CONT_MOBILE
#FLAG_PHONE
#FLAG_EMAIL





















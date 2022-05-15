#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import time
import os
import sys

from IPython.display import display

# Import the SparkSession module
from pyspark.sql import SparkSession
# Import the SQL Context from PySpark SQL
from pyspark.sql import SQLContext

from pyspark.mllib.random import RandomRDDs
import pyspark.sql.functions as F
#from pyspark.sql.functions import col, sqrt, expr, when, rand
from operator import add
from functools import reduce

# Build the SparkSession
spark = SparkSession.builder.appName("PySpark_Test_Skript").getOrCreate()
   
# Main entry point for Spark functionality. A SparkContext represents the
# connection to a Spark cluster, and can be used to create :class:`RDD` and
# broadcast variables on that cluster.      
sc = spark.sparkContext

# Get the SQL Context with the SparkContext Parameter
sqlContext = SQLContext(sc)

from IPython.core.display import HTML
display(HTML("<style>pre { white-space: pre !important; }</style>"))


jobs = ['Data_loading_csv','Data_loading_json', 'Data_loading_parquet', 'Count_per_column', 'Mean_per_column',
        'Sum_per_column', 'Standard_deviation', 'Summary', 'Filter', 'Avg_addition_2_columns', 
        'Sum_addition_2_columns', 'Product_2_columns', 'Add_new_column', 'Add_new_column_complex_calculation', 
        'GroupBy', 'Distinct', 'Join']

dataframe_dimensions = ['1/1', '1/10', '1/100', '1/1000', '1/10000', '1/100000', '1/1000000', '1/10000000',
                   '5/1', '5/10', '5/100', '5/1000', '5/10000', '5/100000', '5/1000000', '5/10000000',
                   '10/1', '10/10', '10/100', '10/1000', '10/10000', '10/100000', '10/1000000', '10/10000000',
                   '20/1', '20/10', '20/100', '20/1000', '20/10000', '20/100000', '20/1000000', '20/10000000',
                   '30/1', '30/10', '30/100', '30/1000', '30/10000', '30/100000', '30/1000000', '30/10000000',
                   '40/1', '40/10', '40/100', '40/1000', '40/10000', '40/100000', '40/1000000', '40/10000000',]

time_df = pd.DataFrame({'Data_loading_csv':pd.Series(dtype='float'),
                        'Data_loading_json':pd.Series(dtype='float'),
                        'Data_loading_parquet':pd.Series(dtype='float'),
                        'Count_per_column':pd.Series(dtype='float'),
                        'Mean_per_column':pd.Series(dtype='float'),
                        'Median_per_column':pd.Series(dtype='float'),
                        'Max_per_column':pd.Series(dtype='float'),
                        'Min_per_column':pd.Series(dtype='float'),
                        'Sum_per_column':pd.Series(dtype='float'),
                        'Standard_deviation_per_column':pd.Series(dtype='float'),
                        'Summary':pd.Series(dtype='float'),
                        'Filter':pd.Series(dtype='float'),
                        'Avg_addition_2_columns':pd.Series(dtype='float'),
                        'Sum_addition_2_columns':pd.Series(dtype='float'),
                        'Product_addition_2_columns':pd.Series(dtype='float'),
                        'Add_new_column':pd.Series(dtype='float'),
                        'Add_new_column_comparing_size':pd.Series(dtype='float'),
                        'GroupBy':pd.Series(dtype='float'),
                        'Distinct':pd.Series(dtype='float'),
                        'Number_distinct_values':pd.Series(dtype='float'),
                        'Join_raw':pd.Series(dtype='float'),
                        'Join':pd.Series(dtype='float'),
                        }, index=dataframe_dimensions)


PATH = './Dataframes/'

df_support = spark.read.csv("Support_Dataframe_6_1000.csv", header=True) 


def time_measurement(function, df, function_name, dataframe_dimension, nr_repetitions=1, **kwargs):
    measured_times = []
    for i in range(nr_repetitions):
        start_time = time.time()
        ret = function(df, **kwargs)
        measured_times.append(time.time()-start_time)
    time_df.loc[dataframe_dimension, function_name] = np.mean(measured_times)
    print(f"{function_name} - {dataframe_dimension} - Time: {time_df.loc[dataframe_dimension, function_name]}")
    return ret
    
def Data_loading_csv (df, file_name):
    return spark.read.csv(PATH + file_name, header=True) 
    
def Data_loading_parquet(df, file_name):
    return spark.read.parquet(PATH + file_name)

def Data_loading_json(df, file_name):
    return spark.read.json(PATH + file_name, multiLine=True)

def Count_per_column(df):
    return df.select([F.count(F.when(F.col(c).isNotNull(), c)).alias(c) for c in df.columns])

def Mean_per_column(df):
    return df.select([F.mean(c).alias(c) for c in df.columns])

def Median_per_column(df):
    return df.select([F.percentile_approx(c, 0.5).alias(c) for c in df.columns])

def Max_per_column(df):
    return df.select([F.max(c).alias(c) for c in df.columns])

def Min_per_column(df):
    return df.select([F.min(c).alias(c) for c in df.columns])

def Sum_per_column(df):
    return df.select([F.sum(c).alias(c) for c in df.columns])

def Standard_deviation_per_column(df):
    return df.select([F.stddev(c).alias(c) for c in df.columns])

def Summary(df):
    return df.summary()

def Filter(df, column, upperbound, lowerbound):
    return df.filter((F.col(column) > lowerbound)&(F.col(column) < upperbound))

def Avg_addition_2_columns(df, column_1, column_2):
    return df.select(((F.col(column_1) + F.col(column_2))/2))#.alias("Avg_addition_2_columns"))

def Sum_addition_2_columns(df, column_1, column_2):
    return df.select((F.col(column_1) + F.col(column_2)))#.alias("Sum_addition_2_columns"))

def Product_addition_2_columns(df, column_1, column_2):
    return df.select((F.col(column_1) * F.col(column_2)))#.alias("Product_addition_2_columns"))

def Add_new_column(df, df_2, column_1, column_2, column_name): 
    df_2 = df_2.withColumn(column_name, F.col(column_1) + F.col(column_2))
    return df_2

def Add_new_column_comparing_size(df, df_2, column_1, column_2, column_name):
    df_2 = df_2.withColumn(column_name, F.when((F.col(column_1) > F.col(column_2)), 1).when((F.col(column_1) < F.col(column_2)), 2).otherwise("Tie"))
    return df_2

def GroupBy(df, column):
    return df.groupBy(column).agg(*[F.sum(c).alias(c) for c in df.columns if c != column])

def Distinct(df, column):
    return df.select(column).distinct()

def Number_distinct_values(df):
    return df.select([F.countDistinct(c).alias(c) for c in df.columns])

def Join_raw(df_1, df_2):
    return df_1.join(df_2, df_1.id == df_2.id, "left").drop(df_2.id)

def Join(df_1, df_2, column):
    df_supp_1 = df_1.groupBy(column).agg(*[F.sum(c).alias(c) for c in df_1.columns if c != column])
    df_supp_2 = df_2.groupBy(column).agg(*[F.sum(c).alias(c) for c in df_2.columns if c != column])
    return df_supp_1.join(df_supp_2, df_supp_1.id == df_supp_2.id, "left").drop(df_supp_2.id)


for i in dataframe_dimensions:
    x,y = i.split('/')
    
    time_measurement(Data_loading_csv, None, 'Data_loading_csv', i, 1, file_name=f"Dataframe_{x}_{y}.csv")
    df = time_measurement(Data_loading_parquet, None, 'Data_loading_parquet', i, 1, file_name=f"Dataframe_{x}_{y}.parquet.gz")
    time_measurement(Data_loading_json, None, 'Data_loading_json', i, 1, file_name=f"Dataframe_{x}_{y}.json.gz")

    df_2 = df

    time_measurement(Count_per_column, df, 'Count_per_column', i)
    time_measurement(Mean_per_column, df, 'Mean_per_column', i)
    time_measurement(Median_per_column, df, 'Median_per_column', i)
    time_measurement(Max_per_column, df, 'Max_per_column', i)
    time_measurement(Min_per_column, df, 'Min_per_column', i)
    time_measurement(Sum_per_column, df, 'Sum_per_column', i)
    
    time_measurement(Standard_deviation_per_column, df, 'Standard_deviation_per_column', i)
    time_measurement(Summary, df, 'Summary', i)
    time_measurement(Filter, df, 'Filter', i, column='col0', upperbound=80, lowerbound=-40)
    
    if x == '1':
        time_measurement(Avg_addition_2_columns, df, 'Avg_addition_2_columns', i, column_1='col0', column_2='col0')
        time_measurement(Sum_addition_2_columns, df, 'Sum_addition_2_columns', i, column_1='col0', column_2='col0')

        time_measurement(Product_addition_2_columns, df, 'Product_addition_2_columns', i, column_1='col0', column_2='col0')
        time_measurement(Add_new_column, df, 'Add_new_column', i, df_2=df_2, column_1='col0', column_2='col0', column_name = 'Add_col')
        time_measurement(Add_new_column_comparing_size, df, 'Add_new_column_comparing_size', i, df_2=df_2, column_1='col0', column_2='col0', column_name = 'Comp_Size')
        time_measurement(GroupBy, df, 'GroupBy', i, column='id')
        time_measurement(Distinct, df, 'Distinct', i, column='col0')
    else:   
        time_measurement(Avg_addition_2_columns, df, 'Avg_addition_2_columns', i, column_1='col2', column_2='col3')
        time_measurement(Sum_addition_2_columns, df, 'Sum_addition_2_columns', i, column_1='col2', column_2='col3')

        time_measurement(Product_addition_2_columns, df, 'Product_addition_2_columns', i, column_1='col2', column_2='col3')
        time_measurement(Add_new_column, df, 'Add_new_column', i, df_2=df_2, column_1='col2', column_2='col3', column_name = 'Add_col')
        time_measurement(Add_new_column_comparing_size, df, 'Add_new_column_comparing_size', i, df_2=df_2, column_1='col2', column_2='col3', column_name = 'Comp_Size')
        time_measurement(GroupBy, df, 'GroupBy', i, column='id')
        time_measurement(Distinct, df, 'Distinct', i, column='col2')
        
    time_measurement(Number_distinct_values, df, 'Number_distinct_values', i)
    time_measurement(Join_raw, df, 'Join_raw', i, df_2=df_support)
    time_measurement(Join, df, 'Join', i, df_2=df_support, column='id')

display(time_df)
time_df.to_csv("./Desktop/New_Version/Results/Cluster_PySpark_Result.csv")







import os
import pyspark
from pyspark import SparkContext

from dateutil import parser

from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.sql.types import DoubleType
from pyspark.sql import functions as F
from pyspark.sql.functions import udf

from pyspark.sql.functions import isnan, when, count, col

from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
from pyspark.sql.functions import lit

from pyspark.sql import SQLContext
sqlContext=SQLContext(spark.sparkContext, sparkSession=spark, jsqlContext=None)

import re

from pyspark.ml.feature import HashingTF, IDF, RegexTokenizer, StringIndexer, Word2Vec
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics
from fuzzywuzzy import fuzz
import string

spark = SparkSession \
		.builder \
		.appName("Big data project") \
		.config("spark.some.config.option", "some-value") \
		.getOrCreate()

def getData(file):
    return spark.read.option("delimiter", "\\t").option("inferSchema", "true").option("header","true").csv(file, inferSchema=True)


def getDataCustom(file):
    return spark.read.option("delimiter", ",").option("inferSchema", "true").option("header","true").csv(file, inferSchema=True)


with open("/home/jys308/cluster1.txt","r") as f:
	content = f.readlines()

files = content[0].strip("[]").replace("'","").replace(" ","").split(",")

########################################
# Main semantic type function
########################################

def semanticType(colName, df):
    """
    The semantic types:
    1) Person name (Last name, First name, Middle name, Full name) NAME
    2) Business name NAME
    3) Phone Number REGEX
    4) Address REGEX
    5) Street name NAME
    6) City LEVEN
    7) Neighborhood LEVEN
    8) LAT/LON coordinates REGEX
    9) Zip code REGEX
    10) Borough LEVEN
    11) School name (Abbreviations and full names) LEVEN
    12) Color LEVEN
    13) Car make LEVEN
    14) City agency (Abbreviations and full names) LEVEN
    15) Areas of study (e.g., Architecture, Animal Science, Communications) LEVEN
    16) Subjects in school (e.g., MATH A, MATH B, US HISTORY) LEVEN
    17) School Levels (K-2, ELEMENTARY, ELEMENTARY SCHOOL, MIDDLE) LEVEN
    18) College/University names LEVEN
    19) Websites (e.g., ASESCHOLARS.ORG) REGEX
    20) Building Classification (e.g., R0-CONDOMINIUM, R2-WALK-UP) LEVEN
    21) Vehicle Type (e.g., AMBULANCE, VAN, TAXI, BUS) LEVEN
    22) Type of location (e.g., ABANDONED BUILDING, AIRPORT TERMINAL, BANK, CHURCH, CLOTHING/BOUTIQUE) LEVEN
    23) Parks/Playgrounds (e.g., CLOVE LAKES PARK, GREENE PLAYGROUND) LEVEN

    We will check the column name and it's levenshtein distance with the list of semantic types. We will call the    function for that semantic type.

    input: df with 2 columns. 1st column is the data and 2nd is the count
    output: dictionary with keys as semantic types and values as count
    """

    types = {}



    def REGEX(df):
        types = {}
        ########################
        # There are five types that we will find with regex
        ########################
        web_regex = r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"

        zip_regex = r"11422 11422-7903 11598 11678787 11678-23 11723 11898-111 22222222-6666 14567-999999 11111-2222"
        
        latlong_regex = r'([0-9.-]+).+?([0-9.-]+)'

        phone_regex = r'((\(\d{3}\) ?)|(\d{3}-))?\d{3}-\d{4}'

        #could improve the address regex

        address_regex = r'^\d+\s[A-z]+\s[A-z]+'

        columns = df.columns

        df_web = df.filter(df[columns[0]].rlike(web_regex))
        df_zip = df.filter(df[columns[0]].rlike(zip_regex))
        df_latlong =  df.filter(df[columns[0]].rlike(latlong_regex))
        df_phone =  df.filter(df[columns[0]].rlike(phone_regex))
        df_address =  df.filter(df[columns[0]].rlike(address_regex))

	#Get rows from df_web, which will be webaddress, and sum the second column
	#to get the semantic type WEBSITE.
        #only sum and add type if it exists

	if len(df_web.take(1)) > 0:
	    #not sure which is faster
	    #web_frequency = df_web.rdd.map(lambda x: (1,x[1])).reduceByKey(lambda x,y: x + y).collect()[0][1]
	    web_frequency = df_web.groupBy().sum().collect()[0][0]
	    types['web'] = web_frequency

        if len(df_zip.take(1)) > 0:
            zip_frequency = df_zip.groupBy().sum().collect()[0][0]
            types['zip'] = zip_frequency

        if len(df_latlong.take(1)) > 0:
            latlong_frequency = df_latlong.groupBy().sum().collect()[0][0]
            types['latlong'] = latlong_frequency

        if len(df_phone.take(1)) > 0:
            phone_frequency = df_phone.groupBy().sum().collect()[0][0]
            types['phone'] = phone_frequency

        if len(df_address.take(1)) > 0:
            address_frequency = df_address.groupBy().sum().collect()[0][0]
            types['address'] = address_frequency


        return types

    def NAME(df):
        types = {}
        #take column one and make predictions
        predictions = model.transform(...)
        return types

    def LEVEN(df):
        types = {}
        return types

    types_names = {}
    types_regex = {}
    types_leven = {}

    #check levenshtein distance with NAME
    if (fuzz.partial_ratio(colName.lower(), 'name') > 0.75) or (fuzz.partial_ratio(colName.lower(), 'street') > 0.75):
        types_names = NAME(df)

    types_regex = REGEX(df)

    types_leven = LEVEN(df)

    #merge all three dictionaries

    types = {**types_names, **types_regex, **types_leven}

    return types

#################################
# Train Classifier
#################################

address_df = getDataCustom('/user/jys308/Address_Point.csv')
permit_df = getDataCustom('/user/jys308/Approved_Permits.csv')
address_df.createOrReplaceTempView("address_df")
permit_df.createOrReplaceTempView("permit_df")

full_st_name = spark.sql("SELECT DISTINCT FULL_STREE as TEXT FROM address_df")
st_name = spark.sql("SELECT DISTINCT ST_NAME as TEXT FROM address_df")
first_name = spark.sql("SELECT `Applicant First Name` as TEXT from permit_df")
last_name = spark.sql("SELECT `Applicant Last Name` as TEXT from permit_df")
business_name = spark.sql("SELECT DISTINCT `Applicant Business Name` as TEXT FROM permit_df")

st_name = st_name.withColumn('category', lit('STREETNAME'))
full_st_name = full_st_name.withColumn('category', lit('STREETNAME'))
first_name = first_name.withColumn('category', lit('HUMANNAME'))
last_name = last_name.withColumn('category', lit('HUMANNAME'))
business_name = business_name.withColumn('category', lit('BUSINESSNAME'))

train1 = business_name.union(st_name)
train2 = first_name.union(last_name)
train3 = train1.union(train2)
trainingData =  train3.union(full_st_name)
trainingData = trainingData.dropna()

indexer = StringIndexer(inputCol="category", outputCol="label")
tokenizer = RegexTokenizer(pattern=u'\W+', inputCol="TEXT", outputCol="words", toLowercase=False)
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
idf = IDF(inputCol="rawFeatures", outputCol="features")
lr = LogisticRegression(maxIter=20, regParam=0.001)
pipeline = Pipeline(stages=[indexer, tokenizer, hashingTF, idf, lr])
model = pipeline.fit(trainingData)

#######################################
# Iterate through all columns
#######################################

for file in files:
	fileData = file.split(".")
	fileName = fileData[0]
	colName = fileData[1]
	df = spark.read.option("delimiter", "\\t").option("header","true").option("inferSchema","true").csv("/user/hm74/NYCColumns/" + file)
        types = semanticTypes(colName, df)
        #process dictionary to record to json


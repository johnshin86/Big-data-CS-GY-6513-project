import os
import pyspark
from pyspark import SparkContext

from dateutil import parser

from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.sql.types import DoubleType
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql import Row

from pyspark.sql.functions import isnan, when, count, col

from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
from pyspark.sql.functions import lit
from pyspark.sql.functions import levenshtein  

from pyspark.sql import SQLContext
sqlContext=SQLContext(spark.sparkContext, sparkSession=spark, jsqlContext=None)

import re

from pyspark.ml.feature import HashingTF, IDF, RegexTokenizer, IndexToString, StringIndexer, Word2Vec
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
    1) Person name (Last name, First name, Middle name, Full name) NAME (DONE)
    2) Business name NAME (DONE)
    3) Phone Number REGEX (DONE)
    4) Address REGEX (DONE)
    5) Street name NAME (DONE)
    6) City LEVEN
    7) Neighborhood LEVEN
    8) LAT/LON coordinates REGEX (DONE)
    9) Zip code REGEX (DONE)
    10) Borough LEVEN 
    11) School name (Abbreviations and full names) LEVEN
    12) Color LEVEN
    13) Car make LEVEN
    14) City agency (Abbreviations and full names) LEVEN
    15) Areas of study (e.g., Architecture, Animal Science, Communications) LEVEN
    16) Subjects in school (e.g., MATH A, MATH B, US HISTORY) LEVEN
    17) School Levels (K-2, ELEMENTARY, ELEMENTARY SCHOOL, MIDDLE) LEVEN
    18) College/University names LEVEN
    19) Websites (e.g., ASESCHOLARS.ORG) REGEX (DONE)
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
	columns = df.columns
        #take column one and make predictions

	df= df.select(col(columns[0]).alias("TEXT"), col(columns[1]).alias("frequency"))

        pred = model.transform(df.select('TEXT'))
	pred_categories = pred.select('TEXT', 'originalcategory')
	new_df = df.join(pred_categories, on=['TEXT'], how='left_outer')
	
	street_name_df = new_df.filter(new_df['originalcategory'] == 'STREETNAME')
	human_df = new_df.filter(new_df['originalcategory'] == 'HUMANNAME')
	business_df = new_df.filter(new_df['originalcategory'] == 'BUSINESSNAME')

	if len(street_name_df.take(1)) > 0:
            stname_frequency = street_name_df.groupBy().sum().collect()[0][0]
            types['stname'] = stname_frequency

	if len(human_df.take(1)) > 0:
	    human_frequency = human_df.groupBy().sum().collect()[0][0]
	    types['humanname'] = human_frequency

	if len(business_df.take(1)) > 0:
	    business_frequency = business_df.groupBy().sum().collect()[0][0]
	    types['businessname'] = business_frequency

        return types

    def LEVEN(df):

        ###############
        # Cities
        ###############
        cities_df

        ###############
        # Neighborhoods
        ###############
        neighborhood_df

        ###############
        # Borough
        ###############
        borough_df

        ###############
        # School Name
        ###############
        schoolname_df

        ###############
        # Color
        ###############
        color_df

        ###############
        # Carmake
        ###############
        carmake_df

        ###############
        # City Agency
        ###############
        cityagency_df

        ##############
        # Area of Study
        ##############
        areastudy_df

        ##############
        # Subjects
        ##############
        subjects_df

        ##############
        # School Levels
        ##############
        schoollevels_df

        ##############
        # Colleges
        ##############
        college_df

        ##############
        # Vehicle Type
        ##############
        vehicletype_df

        ##############
        # Type of Location
        ##############
        typelocation_columns = typelocation_df.columns
        typelocation_crossjoin = df.crossJoin(typelocation_df)
        typelocation_levy = typelocation_crossjoin.withColumn("word1_word2_levenshtein",levenshtein(col(typelocation_columns[0]), col('typelocation')))
        typelocation_counts = typelocation_levy.filter(typelocation_levy["word1_word2_levenshtein"] <= 2)
        if len(typelocation_counts.take(1)) > 0:
            typelocation_frequency = typelocation_counts.groupBy().sum().collect()[0][0]
            types['typelocation'] = typelocation_frequency

        ##############
        # Parks
        ##############
        parks_columns = parks_df.columns
        parks_crossjoin = df.crossJoin(park_df)
        parks_levy = parks_crossjoin.withColumn("word1_word2_levenshtein",levenshtein(col(parks_columns[0]), col('parks')))
        park_counts = parks_levy.filter(parks_levy['word1_word2_levenshtein'] <= 2)
        if len(park_counts.take(1)) > 0:
            #will this indexing cause issues if first column is integer schema?
            parks_frequency = park_counts.groupBy().sum().collect()[0][0]
            types['parks'] = parks_frequency


	################
	# Building Codes
	################

	building_columns = building_code_df.columns
	building_crossjoin = df.crossJoin(building_code_df)
	building_code_levy = building_crossjoin.withColumn("word1_word2_levenshtein",levenshtein(col(building_columns[0]), col('building_codes')))
	building_counts = building_code_levy.filter(building_code_levy['word1_word2_levenshtein'] <= 1)
	if len(building_counts.take(1)) > 0:
		building_code_frequency = building_counts.groupBy().sum().collect()[0][0]
		types['building_code'] = building_code_frequency



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

#we could add more data, but this is what we had time for

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
fullData =  train3.union(full_st_name)

#(trainingData, testData) = fullData.randomSplit([0.9, 0.1])

#trainingData = trainingData.dropna()
#testData = testData.dropna()

indexer = StringIndexer(inputCol="category", outputCol="label").fit(trainingData)
tokenizer = RegexTokenizer(pattern=u'\W+', inputCol="TEXT", outputCol="words", toLowercase=False)
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
idf = IDF(inputCol="rawFeatures", outputCol="features")
lr = LogisticRegression(maxIter=20, regParam=0.001)
labelConverter = IndexToString(inputCol="prediction", outputCol="originalcategory", labels=indexer.labels)

pipeline = Pipeline(stages=[indexer, tokenizer, hashingTF, idf, lr, labelConverter])
model = pipeline.fit(fullData)

#pred = model.transform(testData)
#pl = pred.select("label", "prediction").rdd.cache()
#metrics = MulticlassMetrics(pl)
#metrics.fMeasure()

#######################################
# Gathering Data for Levenshtein Distance checking
#######################################

"""
1) City 
2) Neighborhood
3) Borough
4) School Name
5) Color
6) Car Make
7) City Agency
8) Areas of Study
9) Subjects in School
10) School levels.
11) College Universities
12) Building Classification (DONE)
13) Vehicle Type
14) Type of Location
15) Parks/Playgrounds
"""

cities_df = spark.createDataFrame(list(map(lambda x: Row(cities=x), cities_list)))
neighborhood_df = spark.createDataFrame(list(map(lambda x: Row(neighborhood=x), neighborhood_list)))
borough_df = spark.createDataFrame(list(map(lambda x: Row(borough=x), borough_list)))
schoolname_df = spark.createDataFrame(list(map(lambda x: Row(schoolname=x), schoolname_list)))
color_df = spark.createDataFrame(list(map(lambda x: Row(color=x), color_list)))
carmake_df = spark.createDataFrame(list(map(lambda x: Row(carmake=x), carmake_list)))
cityagency_df = spark.createDataFrame(list(map(lambda x: Row(cityagency=x), cityagency_list)))
areastudy_df = spark.createDataFrame(list(map(lambda x: Row(areastudy=x), areastudy_list)))
subjects_df = spark.createDataFrame(list(map(lambda x: Row(subjects=x), subject_list)))
schoollevels_df = spark.createDataFrame(list(map(lambda x: Row(schoollevels=x), schoollevels_list)))
college_df = spark.createDataFrame(list(map(lambda x: Row(college=x), college_list)))
vehicletype_df = spark.createDataFrame(list(map(lambda x: Row(vehicletype=x), vehicletype_list)))
typelocation_df = spark.createDataFrame(list(map(lambda x: Row(typelocation=x), typelocation_list)))
parks_df = spark.createDataFrame(list(map(lambda x: Row(parks=x), parks_list)))

###

building_codes_file = open("/home/jys308/building_codes.txt")
building_codes_list = []

for line in building_codes_file:
    line = line.split()
    building_codes_list.append(line[0])

building_code_df = spark.createDataFrame(list(map(lambda x: Row(building_codes=x), building_codes_list)))

####



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

"""
for file in one_file:
	fileData = file.split(".")
	fileName = fileData[0]
	colName = fileData[1]
	df = spark.read.option("delimiter", "\\t").option("header","true").option("inferSchema","true").csv("/user/hm74/NYCColumns/" + file)
"""

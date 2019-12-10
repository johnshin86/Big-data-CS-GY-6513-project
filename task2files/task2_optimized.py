import os
import os.path
from os import path
import pyspark
from pyspark import SparkContext
import json

from dateutil import parser

from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, FloatType
from pyspark.sql.types import DoubleType
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql import Row

from pyspark.sql.functions import isnan, when, count, col

from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import col
from pyspark.sql.functions import lit
from pyspark.sql.functions import levenshtein

from pyspark.sql import SQLContext

import re

from pyspark.ml.feature import HashingTF, IDF, RegexTokenizer, IndexToString, StringIndexer, Word2Vec
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.mllib.evaluation import MulticlassMetrics
from fuzzywuzzy import fuzz
import string

spark = SparkSession \
		.builder \
		.appName("Big data project") \
		.config("spark.some.config.option", "some-value") \
		.getOrCreate()

sqlContext=SQLContext(spark.sparkContext, sparkSession=spark, jsqlContext=None)

def getData(file):
    return spark.read.option("delimiter", "\\t").option("inferSchema", "true").option("header","true").csv(file, inferSchema=True)


def getDataCustom(file):
    return spark.read.option("delimiter", ",").option("inferSchema", "true").option("header","true").csv(file, inferSchema=True)


with open("/home/jys308/cluster1.txt","r") as f:
    content = f.readlines()

def max_vector(vector):
    max_val = float(max(vector))
    return max_val

max_vector_udf = udf(max_vector, FloatType())

files = content[0].strip("[]").replace("'","").replace(" ","").split(",")

########################################
# Main semantic type function
########################################

def REGEX(df):
    print("computing Regex for:", colName)
    web_regex = r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"
    zip_regex = r"11422 11422-7903 11598 11678787 11678-23 11723 11898-111 22222222-6666 14567-999999 11111-2222"
    latlong_regex = r'([0-9.-]+).+?([0-9.-]+)'
    phone_regex = r'((\(\d{3}\) ?)|(\d{3}-))?\d{3}-\d{4}'

    #could improve the address regex
    address_regex = r'^\d+\s[A-z]+\s[A-z]+'

    columns = df.columns

    df = df.withColumn("true_type", when(df[columns[0]].rlike(web_regex), "WEBSITE").otherwise(df["true_type"]))
    df = df.withColumn("true_type", when(df[columns[0]].rlike(zip_regex), "ZIPCODE").otherwise(df["true_type"]))
    df = df.withColumn("true_type", when(df[columns[0]].rlike(latlong_regex), "LATLONG").otherwise(df["true_type"]))
    df = df.withColumn("true_type", when(df[columns[0]].rlike(phone_regex), "PHONENUMBER").otherwise(df["true_type"]))
    df = df.withColumn("true_type", when(df[columns[0]].rlike(address_regex), "ADDRESS").otherwise(df["true_type"]))
    return df

def NAME(df):
    print("computing names for:", colName)
    columns = df.columns
    #take column one and make predictions
    df = df.select(col(columns[0]).alias("TEXT"), col(columns[1]).alias("frequency"), col("true_type"))
    pred = model.transform(df.select('TEXT'))
    pred_categories = pred.select('TEXT', 'originalcategory', 'probability') # add probability vector
    new_df = df.join(pred_categories, on=['TEXT'], how='left_outer')
    # create new DF, filter only for high probability.
    new_df = new_df.withColumn("max_probability", max_vector_udf(new_df["probability"]))
    #new_df.select("probability").map(lambda x: x.toArray().max())
    new_df = new_df.withColumn("true_type", when(new_df["max_probability"] >= 0.95,
    new_df["originalcategory"]).otherwise(new_df["true_type"]))
    df = new_df.drop("originalcategory").drop("probability").drop("max_probability")
    return df

def leven_helper(df, ref_df, cut_off):
    df_columns = df.columns
    # grab the non typed entries in the input df
    new_df = df.filter(df["true_type"].isNull())

    ref_columns = ref_df.columns
    crossjoin_df = new_df.crossJoin(ref_df)
    levy_df = crossjoin_df.withColumn("word1_word2_levenshtein",levenshtein(col(df_columns[0]), col(ref_columns[0])))
    count_df =  levy_df.filter(levy_df["word1_word2_levenshtein"] <= cut_off)

    count_columns = count_df.columns
    count_df = count_df.select(col(count_columns[0]).alias("text_field"), col(count_columns[1]).alias("freq_field"), col(count_columns[2]).alias("type_field"))
    count_columns = count_df.columns

    df = df.join(count_df, df[df_columns[0]] == count_df[count_columns[0]], 'left') #join with low lev distance rows
    df = df.withColumn("true_type", when(df["type_field"].isNotNull(), "CITY").otherwise(df["true_type"]))
    df = df.drop("text_field").drop("freq_field").drop("type_field")
    return df

def LEVEN(df):
    print("Computing Levenshtein for:", colName)
    df = leven_helper(df, cities_df, 2)
    df = leven_helper(df, neighborhood_df, 2)
    df = leven_helper(df, borough_df, 2)
    df = leven_helper(df, schoolname_df, 2)
    df = leven_helper(df, color_df, 2)
    df = leven_helper(df, carmake_df, 2)
    df = leven_helper(df, cityagency_df, 2)
    df = leven_helper(df, areastudy_df, 2)
    df = leven_helper(df, subjects_df, 2)
    df = leven_helper(df, schoollevels_df, 1)
    df = leven_helper(df, college_df, 2)
    df = leven_helper(df, vehicletype_df, 2)
    df = leven_helper(df, typelocation_df, 1)
    df = leven_helper(df, parks_df, 1)
    df = leven_helper(df, building_code_df, 1)
    return df


def semanticType(colName, df):
    df = NAME(df)
    df = REGEX(df)
    df = LEVEN(df)
    return df

#################################
# Train Classifier
#################################


if path.exists("weights"):
    print("Found weights, loading them...")
    model = PipelineModel.load("weights")

else:
    print("Training Classifier...")

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
    fullData = fullData.dropna()

    #(trainingData, testData) = fullData.randomSplit([0.9, 0.1])

    #trainingData = trainingData.dropna()
    #testData = testData.dropna()

    indexer = StringIndexer(inputCol="category", outputCol="label").fit(fullData)
    tokenizer = RegexTokenizer(pattern=u'\W+', inputCol="TEXT", outputCol="words", toLowercase=False)
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    lr = LogisticRegression(maxIter=20, regParam=0.001)
    labelConverter = IndexToString(inputCol="prediction", outputCol="originalcategory", labels=indexer.labels)

    pipeline = Pipeline(stages=[indexer, tokenizer, hashingTF, idf, lr, labelConverter])
    model = pipeline.fit(fullData)

    print("Done training classifier")

    model.save("/home/jys308/weights")

    #pred = model.transform(testData)
    #pl = pred.select("label", "prediction").rdd.cache()
    #metrics = MulticlassMetrics(pl)
    #metrics.fMeasure()

#######################################
# Gathering Data for Levenshtein Distance checking
#######################################

print("Loading text files...")

###
# Cities
###
cities_file = open("/home/jys308/cities.txt")
cities_list = []

for line in cities_file:
    #line = line.split()
    cities_list.append(line[:-1])

cities_df = spark.createDataFrame(list(map(lambda x: Row(cities=x), cities_list)))

###
# Neighborhood
###

neighborhood_file = open("/home/jys308/neighborhood.txt")
neighborhood_list = []

for line in neighborhood_file:
    #line = line.split()
    neighborhood_list.append(line[:-1])

neighborhood_df = spark.createDataFrame(list(map(lambda x: Row(neighborhood=x), neighborhood_list)))

###
# Borough
###

borough_file = open("/home/jys308/boroughs.txt")
borough_list = []

for line in borough_file:
    #line = line.split()
    borough_list.append(line[:-1])

borough_df = spark.createDataFrame(list(map(lambda x: Row(borough=x), borough_list)))

###
# Schoolname
###

schoolname_file = open("/home/jys308/schoolname.txt")
schoolname_list = []

for line in schoolname_file:
    #line = line.split()
    schoolname_list.append(line[:-1])

schoolname_df = spark.createDataFrame(list(map(lambda x: Row(schoolname=x), schoolname_list)))

###
# Color
###

color_file = open("/home/jys308/colors.txt")
color_list = []

for line in color_file:
    #line = line.split()
    color_list.append(line[:-1])

color_df = spark.createDataFrame(list(map(lambda x: Row(color=x), color_list)))

###
# Carmake
###

carmake_file = open("/home/jys308/carmake.txt")
carmake_list = []

for line in carmake_file:
    #line = line.split()
    carmake_list.append(line[:-1])

carmake_df = spark.createDataFrame(list(map(lambda x: Row(carmake=x), carmake_list)))

###
# City Agency
###

cityagency_file = open("/home/jys308/cityagency.txt")
cityagency_list = []

for line in cityagency_file:
    #line = line.split()
    cityagency_list.append(line[:-1])

cityagency_df = spark.createDataFrame(list(map(lambda x: Row(cityagency=x), cityagency_list)))

###
# Area Study
###

areastudy_file = open("/home/jys308/areastudy.txt")
areastudy_list = []

for line in areastudy_file:
    #line = line.split()
    areastudy_list.append(line[:-1])

areastudy_df = spark.createDataFrame(list(map(lambda x: Row(areastudy=x), areastudy_list)))

###
# Subjects
###

subjects_file = open("/home/jys308/subjects.txt")
subjects_list = []

for line in subjects_file:
    #line = line.split()
    subjects_list.append(line[:-1])

subjects_df = spark.createDataFrame(list(map(lambda x: Row(subjects=x), subjects_list)))

###
# School Levels
###

schoollevels_file = open("/home/jys308/schoollevels.txt")
schoollevels_list = []

for line in schoollevels_file:
    #line = line.split()
    schoollevels_list.append(line[:-1])

schoollevels_df = spark.createDataFrame(list(map(lambda x: Row(schoollevels=x), schoollevels_list)))

###
# College
###

college_file = open("/home/jys308/college.txt", encoding="utf-8")
college_list = []

for line in college_file:
    #line = line.split(',')
    #if line[0] == 'US':
    college_list.append(line[:-1])

college_df = spark.createDataFrame(list(map(lambda x: Row(college=x), college_list)))

###
# Vehicle Type
###

vehicletype_file = open("/home/jys308/vehicletype.txt")
vehicletype_list = []

for line in vehicletype_file:
    #line = line.split()
    vehicletype_list.append(line[:-1])

vehicletype_df = spark.createDataFrame(list(map(lambda x: Row(vehicletype=x), vehicletype_list)))

###
# Type Location
###
typelocation_file = open("/home/jys308/typelocation.txt")
typelocation_list = []

for line in typelocation_file:
    line = line[3:-1]
    typelocation_list.append(line)

typelocation_df = spark.createDataFrame(list(map(lambda x: Row(typelocation=x), typelocation_list)))

###
# Parks
###

parks_file = open("/home/jys308/parks.txt")
parks_list = []

for line in parks_file:
    #line = line.split()
    parks_list.append(line[:-1])

parks_df = spark.createDataFrame(list(map(lambda x: Row(parks=x), parks_list)))

###
# Building Codes
###

building_codes_file = open("/home/jys308/building_codes.txt")
building_codes_list = []

for line in building_codes_file:
    line = line.split()
    building_codes_list.append(line[0])

building_code_df = spark.createDataFrame(list(map(lambda x: Row(building_codes=x), building_codes_list)))

####

print("Done Loading Text Files")

#######################################
# Iterate through all columns
#######################################

length = [5308, 10720, 1925, 480, 4, 2, 8, 21520, 788, 6553, 192, 4, 8532, 7062, 12126, 47, 6242, 1693, 6983, 1848, 4, 1797, 19, 71, 4, 2508, 1500, 3111, 2488, 1236, 723, 332, 5965, 4, 17, 7342, 4008, 74, 1543, 1564, 2013, 23730, 2146, 381, 5701, 3187, 6511, 44, 11641, 4, 14899, 5146, 31270, 11, 43, 23, 3, 1608, 1603, 700, 4, 4, 7, 4, 1727, 7, 20958, 52535, 1740, 11, 8, 323, 15045, 21, 11, 1734, 45, 5, 471, 116, 14361, 862, 37, 4103, 1603, 5701, 8, 21, 3, 6, 19, 3, 65, 4, 2364, 5933, 7, 3538, 26, 20, 975, 3976, 21, 432, 1638, 2872, 20007, 689, 9638, 950, 10, 44, 196, 4, 19, 1478, 107, 1701, 32, 38, 152, 2372, 1821, 1200, 20941, 6344, 8771, 4845, 73, 13, 6322, 2895, 558170, 17252, 18032, 479, 4, 846, 8, 4, 20473, 7, 68, 118, 4, 36, 319, 23, 22640, 18636, 38, 4, 4, 1249, 3672, 1518, 4, 10597, 14, 2165, 4, 1755, 27, 6602, 6872, 723, 943, 45, 510, 27, 4292, 1734, 1419, 122763, 7731, 8060, 30, 8657, 15841, 9837, 4150, 6, 5383, 1162, 3744, 190, 2284, 14667, 2532, 2597, 28, 1739, 1733, 43, 1060, 260972, 6031, 384841, 46, 1105, 5, 14035, 1452, 944, 7, 6983, 21, 6, 1630, 1867, 1296, 3, 85, 7921, 13632, 4, 358, 4, 653, 4, 4, 482235, 7935, 1648, 1548, 41441, 1644, 4, 208, 17, 3192, 12177, 11, 2, 1654, 2, 241, 14, 4, 11645, 5, 12082, 2, 35, 375499, 2496, 52, 13063, 3412, 1074, 2369, 1041, 4668, 48, 360, 190, 4, 62, 19, 4, 27, 1848, 6031, 23, 6345, 57, 2577, 1569, 2679, 4892, 1565, 61]

files_and_length = []

for (i,j) in zip(files, length):
    files_and_length.append((i,j))

files_and_length.sort(key = lambda x: x[1])


for file in files_and_length[:10]:
    print("This is the index of current column in sorted list::", files_and_length.index(file))
    file = file[0]
    fileData = file.split(".")
    fileName = fileData[0]
    colName = fileData[1]
    df = spark.read.option("delimiter", "\\t").option("header","true").option("inferSchema","true").csv("/user/hm74/NYCColumns/" + file)
    colNamesList = [i.replace(".","").replace(" ","_") for i in df.columns] # . and space not supported in column name by pyspark
    df = df.toDF(*colNamesList) #change name in DF
    #colName = colNamesList[0] #change colName we have
    colName = colName.replace(".","").replace(" ", "_")

    ###
    # Add third column to df with null types, will amend as we go along.
    ##
    df = df.withColumn('true_type', lit(None))

    df = semanticType(colName, df)

    df_columns = df.columns

    dictionary_df = df.groupBy("true_type").collect()
    print(dictionary_df)

    #print("Working on", colName)
    #print("This is column number", files.index(file))
    #process dictionary to record to json
    #with open(str(file) +'.json', 'w') as fp:
    #    json.dump(types, fp)


#largest file index is 132


"""

files = files[:1]
for file in files:
    fileData = file.split(".")
    fileName = fileData[0]
    colName = fileData[1]
    df = spark.read.option("delimiter", "\\t").option("header","true").option("inferSchema","true").csv("/user/hm74/NYCColumns/" + file)
    print("Working on", colName)
    print("This is column number", files.index(file))
    types = semanticType(colName, df)
    with open('semantic_file_test.txt', 'w') as file:
        file.write(json.dumps(types))
"""

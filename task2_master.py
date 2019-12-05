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

def semanticType(df):
    types = {}
    """
    The semantic types:
    1) Person name
    2) Business name
    3) Phone Number
    4) Address
    5) Street name
    6) City
    7) Neighborhood
    8) LAT/LON coordinates
    9) Zip code
    10) Borough
    11) School name (Abbreviations and full names)
    12) Color
    13) Car make
    14) City agency (Abbreviations and full names)
    15) Areas of study (e.g., Architecture, Animal Science, Communications)
    16) Subjects in school (e.g., MATH A, MATH B, US HISTORY)
    17) School Levels (K-2, ELEMENTARY, ELEMENTARY SCHOOL, MIDDLE)
    18) College/University names
    19) Websites (e.g., ASESCHOLARS.ORG)
    20) Building Classification (e.g., R0-CONDOMINIUM, R2-WALK-UP)
    21) Vehicle Type (e.g., AMBULANCE, VAN, TAXI, BUS)
    22) Type of location (e.g., ABANDONED BUILDING, AIRPORT TERMINAL, BANK, CHURCH, CLOTHING/BOUTIQUE)
    23) Parks/Playgrounds (e.g., CLOVE LAKES PARK, GREENE PLAYGROUND)

    Dataframe will be 2 columns. First column is the data and 2nd is the count.
    """



    return types

Column_Names = []

for file in files:
    fileData = file.split(".")
    colName = fileData[1]
    Column_Names.append(colName)

for file in files:
	fileData = file.split(".")
	fileName = fileData[0]
	colName = fileData[1]
	df = spark.read.option("delimiter", "\\t").option("header","true").option("inferSchema","true").csv("/user/hm74/NYCColumns/" + file)



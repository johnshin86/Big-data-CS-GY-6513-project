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


with open("/home/jys308/cluster1.txt","r") as f:
	content = f.readlines()

files = content[0].strip("[]").replace("'","").replace(" ","").split(",")

def semanticType(df):
    types = {}
    # 

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
	


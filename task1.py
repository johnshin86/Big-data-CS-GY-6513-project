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

sc = SparkContext()

spark = SparkSession \
		.builder \
		.appName("Big data project") \
		.config("spark.some.config.option", "some-value") \
		.getOrCreate()

def getFileList():
	op = os.popen("hdfs dfs -ls /user/hm74/NYCOpenData").read().split('\n')
	files = []
	for file in op:
		if '.gz' in file:
			files.append(file.split()[-1])
	return files

def getData(file):
	return spark.read.option("delimiter", "\\t").option("header","true").option("inferSchema","true").csv(file)

def castAsType(x):
	try:
		if float(x).is_integer() == True:
			x = int(x)
			x = 'INTEGER'
			return x
	except:
		pass
	try:
		x = float(x)
		x = "REAL"
		return x
	except:
		try:
			x = parser.parse(x)
			x = "DATE/TIME"
			return x
		except:
			x = str(x)
			x = 'TEXT'
			return x

def getIntStats(rdd):
	subset = rdd.filter(lambda x:x[1] == "INTEGER").map(lambda x:x[0])
	if subset.isEmpty():
		return 0, 0, 0, 0, 0
	else:
		count = subset.countApprox(timeout=10000)
		return count, subset.max(), subset.min(), subset.mean(), subset.stdev()

def getRealStats(rdd):
	subset = rdd.filter(lambda x:x[1] == "REAL").map(lambda x:x[0])
	if subset.isEmpty():	return 0, 0, 0, 0, 0
	count = subset.countApprox(timeout=10000)
	return count, subset.max(), subset.min(), subset.mean(), subset.stdev()

def getDateStats(rdd):
	subset = rdd.filter(lambda x: x[1]=="DATE/TIME").map(lambda x: parse(x[0]))
	if subset.isEmpty(): return 0, '', ''
	count = subset.countApprox(timeout=10000)
	return count, subset.max(), subset.min()

def getTextStats(rdd):
	subset = rdd.filter(lambda x: x[1]=="TEXT").map(lambda x: (x[0],len(x[0])))
	if subset.isEmpty():	return 0, [], [], 0 
	count = subset.countApprox(timeout=10000)
	longest = subset.takeOrdered(5, key = lambda x: -x[1])
	shortest = subset.takeOrdered(5, key= lambda x: x[1])
	avgLen = subset.map(lambda x: x[1]).mean()
	return count, longest, shortest, avgLen


files = getFileList()

#Processing should be done inside for loop for each dataset
for file in files:
	
	df = getData(file).cache()
	rdd = df.rdd.cache()
	

	# 1 & 2
	emptyDf = df.select([count(when(isnan(c) | col(c).contains('NA') | col(c).contains('NULL') | col(c).isNull(),c)).alias(c) for c in df.columns])
	rows = rdd.countApprox(timeout=10000)
	emptyCount = emptyDf.collect()[0]
	nonEmptyCount = {}
	for c in emptyDf.columns:
		nonEmptyCount[c] = rows - emptyCount[c]


	#3 & 4 & 5
	mostFrequent = {}
	distinct = {}
	dataTypes = dict(df.dtypes)
	colStats = {}
	for col in df.columns:
		grouped = df.groupBy(col).count()
		distinctCount = grouped.count()
		tempFreq = grouped.sort(F.desc('count')).select(col).take(5)
		freqList = []
		for item in tempFreq:
			freqList.append(item[0])

		mostFrequent[col] = freqList
		distinct[col] = distinctCount
	
		#Dont need count, already calculated above
		if dataTypes[col] == "string":
			colRdd = df.select(col).dropna().rdd.map(lambda x: x[0])
			colRdd = colRdd.map(lambda x: (x,castAsType(x)))
			intStats = getIntStats(colRdd)
			realStats = getRealStats(colRdd)
			dateStats = getDateStats(colRdd)
			textStats = getTextStats(colRdd)
			colStats[col] = [intStats,realStats,dateStats,textStats]
		elif dataTypes[col] in ["int","long","bigint"]:
			colRdd = df.select(col).dropna().rdd.map(lambda x: x[0])
			count = colRdd.countApprox(timeout=10000)
			intStats = (count, colRdd.max(), colRdd.min(), colRdd.mean(), colRdd.stdev())
			realStats = (0,0,0,0,0)
			dateStats = (0,0,0)
			textStats = (0,0,0,0)
			colStats[col] = [intStats,realStats,dateStats,textStats]
		elif dataTypes[col] in ["double","float"]:
			colRdd = df.select(col).dropna().rdd.map(lambda x: x[0])
			count = colRdd.countApprox(timeout=10000)
			intStats = (0,0,0,0,0)
			realStats = (count, colRdd.max(), colRdd.min(), colRdd.mean(), colRdd.stdev())
			dateStats = (0,0,0)
			textStats = (0,0,0,0)
			colStats[col] = [intStats,realStats,dateStats,textStats]


		colRdd.unpersist()

	df.unpersist()
	rdd.unpersist()
import csv
import json
import glob
import pyspark
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth

#Combine json
#Get data types for counting
#Get data types for pattern mining

#(id, list) -> FPGrowth

sc = SparkContext()

spark = SparkSession \
		.builder \
		.appName("Big data project") \
		.config("spark.some.config.option", "some-value") \
		.getOrCreate()

fpData = []
id = 0
for file in glob.glob("p1/*.json"):
	print(file)
	with open(file,'r',encoding='utf-8') as f:
		data = json.load(f)
		for column in data["columns"]:
			#print(column['column_name'])
			types = []
			id+=1
			dataTypes = column["dataTypes"]
			for dataType in dataTypes:
				types.append(dataType["type"])
			#print(types)
			fpData.append((id,types))


freqDf = spark.createDataFrame(fpData,["id","dtypes"])
fpGrowth = FPGrowth(itemsCol="dtypes",minSupport=0.001)
model = fpGrowth.fit(freqDf)
freqSet = model.freqItemsets.collect()

with open('freqDataTypes.csv','w') as f:
	wr = csv.writer(f)
	for item in freqSet:
		if len(item[0]) > 1:
			wr.writerow(item)

with open('task1VizStats.csv','w') as f:
	wr = csv.writer(f)
	for item in fpData:
		wr.writerow(item[1])


read_files = glob.glob("p1/*.json")
with open("task1.json", "w") as outfile:
    outfile.write('[{}]'.format(','.join([open(f, "r").read() for f in read_files])))

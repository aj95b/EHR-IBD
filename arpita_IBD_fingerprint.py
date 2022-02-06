# Databricks notebook source
import pyspark.sql.functions as F
from pyspark.sql.functions import size
import json
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import numpy as np
from sklearn.metrics import * #for silhouette score
import umap.umap_ as umap
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# COMMAND ----------

# MAGIC %pip install umap-learn

# COMMAND ----------

##Failed Experimment with QW's cohort. Very few distinct patients and none with inner join with hadlock_ml_labs_by_patient
ibd_pat = spark.sql("""
SELECT * 
FROM rdp_phi_sandbox.qw3_ibd_cohort
""")
ibd_pat.head()
ibd_pat.createOrReplaceTempView("ibd_pat")
labs = spark.sql("""
SELECT * 
FROM rdp_phi_sandbox.hadlock_procedure_orders
""")
ibd_labs = ibd_pat.join(labs,['pat_id'],how='inner')
ibd_labs.createOrReplaceTempView("ibd_labs")
ibd_labs_filtered = spark.sql("""
SELECT * 
FROM ibd_labs
WHERE result_value != null
""")
ibd_labs = ibd_labs.limit(500000)
ibd_df = ibd_labs.toPandas()
ibd_df = ibd_df.loc[:,~ibd_df.columns.duplicated()]
ibd_df = ibd_df.groupby(['pat_id'],as_index=False).first()
#ibd_data = dict(zip(ibd_df['pat_id'].astype(str), ibd_df[:]))
ibd_df = ibd_df.set_index('pat_id')
#IBD_final_test = IBD_final.iloc[0:3,0:3]
ibd_data = ibd_df.to_dict("index")
ibd_data = {k: v for k, v in ibd_data.items() if k == k}

# COMMAND ----------

#With Yeon Mi's help code to gather people with IBD diagnosis 
cohort_df = spark.sql("SELECT patient_id FROM rdp_phi_sandbox.hadlock_ml_merged_patient")
table_name = 'hadlock_problem_list'
problems_admin_df = spark.sql(
"""
SELECT * FROM rdp_phi_sandbox.{table}
""".format(table=table_name))
mapping_column_names = ['dx_id', 'instance', 'SNOMED_24526004']
table_name = 'hadlock_dx_id_snomed_with_descendants_mapping'
dx_id_snomed_df = spark.sql(
"""
SELECT {columns} FROM rdp_phi_sandbox.{table}
""".format(columns=', '.join(mapping_column_names), table=table_name))
dx_id_snomed_df  = dx_id_snomed_df.withColumn('diagnosis_id', F.concat(F.col('instance'),F.col('dx_id')))
problems_df = problems_admin_df.join(dx_id_snomed_df, ['diagnosis_id'], how='left').join(cohort_df, ['patient_id'], how = 'right').where(F.col('SNOMED_24526004'))
table_name = 'hadlock_encounters'
encounters_admin_df = spark.sql(
"""
SELECT * FROM rdp_phi_sandbox.{table}
""".format(table=table_name))
encounters_admin_df = encounters_admin_df.withColumn('diagnosis_id',F.explode(F.col('diagnosis_id')))
encounters_df = encounters_admin_df.join(dx_id_snomed_df, ['diagnosis_id'], how='left').join(cohort_df, ['patient_id'], how = 'right').where(F.col('SNOMED_24526004'))
problems_df.createOrReplaceTempView("p")  
encounters_df.createOrReplaceTempView("e")
ibd_df = spark.sql("""
SELECT patient_id, 1 as ibd
FROM (SELECT DISTINCT patient_id
FROM e
UNION
SELECT DISTINCT patient_id
FROM p) as ibd
""")

#This table rdp_phi_sandbox.hadlock_ml_labs_by_patient has patient labs with numner of encounters
labs = spark.sql("""
SELECT * 
FROM rdp_phi_sandbox.hadlock_ml_labs_by_patient
""")
ibd_labs = ibd_df.join(labs,['patient_id'],how='inner') 
#ibd_labs.createOrReplaceTempView("il")
#ibd_labs = spark.sql("""
#SELECT * 
#FROM il
#WHERE result_value != null
#""")
ibd_pandas = ibd_labs.toPandas()
ibd_pandas = ibd_pandas.set_index('patient_id')
ibd_data = ibd_pandas.to_dict("index")
ibd_data = {k: v for k, v in ibd_data.items() if k == k}
#print(IBD_final.head())
#ibd_labs_filtered = ibd_labs.filter(size(F.col("encounter_ids")) > 500)
#IBD_final = ibd_labs_filtered.toPandas()
#print(IBD_final.shape)
#IBD_final.head()

#labs for one patient with 500 encounters
#labs_by_enc = spark.sql("""
#SELECT * 
#FROM rdp_phi_sandbox.hadlock_ml_labs_by_encounter
#WHERE patient_id == 10001402397250
#""")

# COMMAND ----------

ibd = ibd_pandas.iloc[:,2:]
for col in ibd.columns:
  if('first' in col or 'last' in col):
    ibd = ibd.drop([col], axis=1)
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(ibd)
principalDf = pd.DataFrame(data = principalComponents, index=ibd.index ,columns = ['PC_1', 'PC_2', 'PC_3'])
plt.scatter(principalDf.iloc[:,0],principalDf.iloc[:,1], s=20)

plt.scatter(principalDf.iloc[y_hc==0,0],principalDf.iloc[y_hc==0,1], s=30, c='red')
plt.scatter(principalDf.iloc[y_hc==1,0],principalDf.iloc[y_hc==1,1], s=30, c='blue')

plt.xlabel('PC_1')
plt.ylabel('PC_2')

# COMMAND ----------

# MAGIC %md
# MAGIC #PCA on all data (removing encounter lists and dates)

# COMMAND ----------

ibd = ibd_pandas.iloc[:,2:]
for col in ibd.columns:
  if('first' in col or 'last' in col or 'neg' in col or 'pos' in col):
    ibd = ibd.drop([col], axis=1)
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(ibd)
principalDf = pd.DataFrame(data = principalComponents, index=ibd.index ,columns = ['PC_1', 'PC_2', 'PC_3'])
plt.scatter(principalDf.iloc[:,0],principalDf.iloc[:,1], s=20)
plt.xlabel('PC_1')
plt.ylabel('PC_2')

# COMMAND ----------

hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage ='ward')
y_hc = hc.fit_predict(principalDf)
plt.scatter(principalDf.iloc[y_hc==0, 0], principalDf.iloc[y_hc==0,1],s=20, c='red', label ='Cluster 1')
plt.scatter(principalDf.iloc[y_hc==1, 0], principalDf.iloc[y_hc==1,1], s=20, c='blue', label ='Cluster 2')
plt.xlabel('PC_1')
plt.ylabel('PC_2')

# COMMAND ----------

clust0 = pd.DataFrame()
clust1 = pd.DataFrame()
#clust2 = pd.DataFrame()
clust0 = fp_df.iloc[y_hc==0, 0:1]
clust1 = fp_df.iloc[y_hc==1, 0:1]
#clust2 = fp_df.iloc[y_hc==2, 0:1]
clust0.reset_index(inplace=True, drop=True)
clust1.reset_index(inplace=True, drop=True)
#clust2.reset_index(inplace=True, drop=True)
clust0 = clust0.rename(columns = {'0':'patient_id'})
clust1 = clust1.rename(columns = {'0':'patient_id'})
#clust2 = clust2.rename(columns = {'0':'patient_id'})
print(clust0.shape)
print(clust1.shape)
#print(clust2.shape)
#clust2
#spark_df_0=spark.createDataFrame(clust0) 
#spark_df_0.write.parquet("abfss://rdp-phi-sandbox@datalakestorage321.dfs.core.windows.net/arpita_ibd_clust0")
#spark_df_1=spark.createDataFrame(clust1) 
#spark_df_1.write.parquet("abfss://rdp-phi-sandbox@datalakestorage321.dfs.core.windows.net/arpita_ibd_clust1")

# COMMAND ----------

reducer = umap.UMAP()
embedding = reducer.fit_transform(ibd)
emb_df = pd.DataFrame(data = embedding, index=ibd.index, columns=['Embedding_1','Embedding_2'])
plt.scatter(emb_df.iloc[:,0],emb_df.iloc[:,1], s=10)
plt.xlabel('UMAP_Embedding_1')
plt.ylabel('UMAP_Embedding_2')

# COMMAND ----------

hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage ='ward')
y_hc = hc.fit_predict(emb_df)
plt.scatter(emb_df.iloc[y_hc==0, 0], emb_df.iloc[y_hc==0,1],s=20, c='red', label ='Cluster 1')
plt.scatter(emb_df.iloc[y_hc==1, 0], emb_df.iloc[y_hc==1,1], s=20, c='blue', label ='Cluster 2')
plt.xlabel('UMAP_Embedding_1')
plt.ylabel('UMAP_Embedding_2')

# COMMAND ----------

clust0 = pd.DataFrame()
clust1 = pd.DataFrame()
#clust2 = pd.DataFrame()
clust0 = fp_df.iloc[y_hc==0, 0:1]
clust1 = fp_df.iloc[y_hc==1, 0:1]
#clust2 = fp_df.iloc[y_hc==2, 0:1]
clust0.reset_index(inplace=True, drop=True)
clust1.reset_index(inplace=True, drop=True)
#clust2.reset_index(inplace=True, drop=True)
clust0 = clust0.rename(columns = {'0':'patient_id'})
clust1 = clust1.rename(columns = {'0':'patient_id'})
#clust2 = clust2.rename(columns = {'0':'patient_id'})
print(clust0.shape)
print(clust1.shape)
#print(clust2.shape)
#clust2
#spark_df_0=spark.createDataFrame(clust0) 
#spark_df_0.write.parquet("abfss://rdp-phi-sandbox@datalakestorage321.dfs.core.windows.net/arpita_ibd_clust0")
#spark_df_1=spark.createDataFrame(clust1) 
#spark_df_1.write.parquet("abfss://rdp-phi-sandbox@datalakestorage321.dfs.core.windows.net/arpita_ibd_clust1")

# COMMAND ----------

ax = plt.axes(projection ="3d")
ax.scatter3D(principalDf.iloc[:, 0], principalDf.iloc[:,1], principalDf.iloc[:,2])
#ax.scatter3D(principalDf.iloc[y_hc==1, 0], principalDf.iloc[y_hc==1,1], principalDf.iloc[y_hc==1,2],s=40, c='blue', label ='Cluster 2')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# COMMAND ----------

hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage ='ward')
y_hc = hc.fit_predict(principalDf)
ax = plt.axes(projection ="3d")
ax.scatter3D(principalDf.iloc[y_hc==0, 0], principalDf.iloc[y_hc==0,1], principalDf.iloc[y_hc==0,2],s=40, c='red', label ='Cluster 1')
ax.scatter3D(principalDf.iloc[y_hc==1, 0], principalDf.iloc[y_hc==1,1], principalDf.iloc[y_hc==1,2],s=40, c='blue', label ='Cluster 2')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# COMMAND ----------

ibd_data1 = ibd.to_dict("index")
ibd_data1 = {k: v for k, v in ibd_data1.items() if k == k}

# COMMAND ----------

################@Author: Arpita Joshi###########################
###########  Modifications to Denise's and Jewel's code  to get one fingerprint vector for each row entry in a JSON object (one file with patients as different rows)######
###########  Modifications to Denise's and Jewel's code  to get:
#1. One fingerprint vector for each row entry in a JSON object (one file with patients as different rows)######
#2. Mantissa encoding: taking into acocunt exponent weight for concordance with Perl version
#3. add_vector_value function rewritten for concordance with Perl version
#4. Bug fix that prevented vector value computation of numbers

#from Json2Vec import *
import sys
import json
import itertools
#from json2fp import *
import pandas as pd
import numpy as np
import math


def isnumeric(n):
	if isinstance(n, float):
		return True
	if isinstance(n, int):
		return True
	#if isinstance(n, str):
		#return False
		
	#else:
		#from re import match
		#return bool(match(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?', str(n)))
	return False




def frexp10(n):
        if n == 0:
            return 0.0, 0
        else:
            e = int(math.log10(abs(n)))
            m = n/10**e
            return m, e

  ## vector_value# compute the value of the first argument in vector form ## 

def vector_value(o):
	length = L
	new = np.zeros(length)

	# for NUMBERS encoding
	if isnumeric(o):
		# if it's null value zero
		if not o:
			return new
			# -----------------------------------------------------------------
			# number method #1: ME (Mantissa/Exponent)
		elif numeric_encoding == "ME":
			mantissa, exponent = frexp10(o)
			# encode mantissa - a fraction in range (-1~1)
			mantissa *= (length / 10.0)  # make mantissa in the range of -L to L
			over = abs(mantissa - int(mantissa))
			exp_w = exponent_weight
			man_w = 1-exp_w
			if over: #> np.finfo(float).eps:
				new[int(mantissa % length)] += man_w*(1 - over)
				index = mantissa + 1 if mantissa > 0 else mantissa - 1
				new[int(index % length)] += man_w*over
			else: # in what case it will come to here?
				new[int(mantissa % length)] += man_w#1
				# encode the exponent - an integer, which can be negative
			new[int(exponent % length)] += exp_w#1

		else:
			new[int(o % length)] += 1

	elif(isinstance(o,(str,bool))):
		
		decay = string_encoding_decay if string_encoding_decay else 0.1
		remain = (1 - decay)
		#print(isinstance(o,str))
		if(isinstance(o,bool)):
			o = str(o)
		v = ord(o[0])
		new[int(v % length)] += 1
		for i in range(1, len(o)):
			v = v*remain + ord(o[i])*decay
			#if self.debug > 5:
				#print("#decay: %s %d %d %.4f" % (o, i, ord(o[i]), v))
			sv = v*length/10.0
			over = sv - int(sv)   
			new[int(sv % length)] += (1-over)
			new[int((sv+1) % length)] += over

	# normalize to a unit vector (sum of 1)
	new = np.array(new) - min(new)
	if sum(new) != 0:
		new = np.array(new) / sum(new)

	return new

'''
 ## add_vector_value## 
def add_vector_value(v1, v2, v3, stuff=None):
        length = L
        for j in range(length):
            v = (v1[j] + v2[int((j+1) % length)] + v3[int((j+2) % length)])/3
            fp[j] += v
'''

def add_vector_value(v1, v2, v3, stuff=None):
	length = L
	tmp = np.zeros(length)
	for i in range(length):
		xx = 1+abs(v1[i] * math.cos(v1[i]))
		yy = 1+abs(v2[i] * math.cos(2*v2[i]))
		zz = 1+abs(v3[i] * math.cos(3*v3[i]))
		tmp[i] = (xx*yy*zz)**(1/3) - 1
	tmp = np.array(tmp) - min(tmp)
	if sum(tmp) != 0:
		tmp = np.array(tmp) / sum(tmp)
	for j in range(length):
		fp[j] = tmp[j]

 ## recurseStructure##
def recurse_structure(obj, name=None, base=None):
	if name is None: name = 'root'
	if base is None: base = vector_value(0)
        # -------------------------------------------------------------------------
        # TYPE 1 data: python dictionary
	global statements
	if isinstance(obj, dict):
		keys_used = 0
		for key, cargo in obj.items():
			#print(cargo)
		# skip empty strings, null value, careful about integer "0" @Arpita: Took care of it in main()
			#if not(cargo): print(cargo)
			if (cargo or isinstance(cargo, int)):
				vkey = vector_value(key)
				if isinstance(cargo, (list, dict)):
			# if it's another list or dict, cargo is the keys_used (length)
					cargo = recurse_structure(cargo, key, vkey)
				add_vector_value(base, vkey, vector_value(cargo),("#hash_entry", name, key, cargo))
				triples.append(list([name, key, cargo]))
				keys_used += 1   # number of statements used in generating this vector
		statements += keys_used
		return keys_used
        
	else:
		return obj


def normalize(temp_fp,key):
	return (np.array(temp_fp)-np.mean(temp_fp)) / np.std(temp_fp)
	

def reset():
	global fp
	global statements
	global triples
	fp = np.zeros(L)
	statements = 0
	triples = []
    
def main():
	row=0
	for key,item in data.items():
		#print(item)
		#fp_vect = np.zeros(L)	
		recurse_structure(item)
		#fp_vect = fp
		global fp
		if(not(np.all((fp == 0)))):
			tfp = normalize(fp,key)
			
		print(key,end='\t')
		fp_df.loc[row,'0'] = key
		print(statements,end='\t')
		fp_df.loc[row,'1'] = int(statements)
		for i in range(len(fp)):
			if i < L-1:
				print(round(tfp[i],decimal),end='\t')
				#fp_df.iloc[row,i+2] = round(tfp[i],decimal)
			
			else: 
				print(round(tfp[i],decimal))
			fp_df.loc[row,i+2] = round(tfp[i],decimal)
		row = row+1				
		reset()

#with open(sys.argv[1]) as f:
#data = json.load(f)
#data = dict(itertools.islice(data.items(), 1))	
#L = int(sys.argv[2])	
#norm = int(sys.argv[3]) @Arpita: Normalization happens anyway
data = ibd_data1
L = 30
root = 'root'
numeric_encoding = 'ME'    # ME
string_encoding = 'decay'  # decay
string_encoding_decay = 0.1
decimal = 3
exponent_weight = 0.5
fp = np.zeros(L)
statements = 0
triples = []
fp_df = pd.DataFrame()
if __name__ == "__main__":
	main()

#spark_df=spark.createDataFrame(fp_df) 
#spark_df.write.parquet("abfss://rdp-phi-sandbox@datalakestorage321.dfs.core.windows.net/arpita_ibd_fingerprint")

# COMMAND ----------

spark_df=spark.createDataFrame(fp_df) 
spark_df.write.parquet("abfss://rdp-phi-sandbox@datalakestorage321.dfs.core.windows.net/arpita_ibd_fingerprint_no_disease_test")

# COMMAND ----------

fp_df = spark.read.parquet("abfss://rdp-phi-sandbox@datalakestorage321.dfs.core.windows.net/arpita_ibd_fingerprint_no_disease_test")
fp_df=fp_df.toPandas()

# COMMAND ----------

fp = fp_df.set_index('0')
fp = fp.iloc[:,1:]
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(fp)
principalDf = pd.DataFrame(data = principalComponents, index=fp.index ,columns = ['PC_1', 'PC_2', 'PC_3'])
#plt.scatter(principalDf.iloc[:,0],principalDf.iloc[:,1], s=20)

hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage ='ward')
y_hc = hc.fit_predict(principalDf)
plt.scatter(principalDf.iloc[y_hc==0,0],principalDf.iloc[y_hc==0,1], s=30, c='red')
plt.scatter(principalDf.iloc[y_hc==1,0],principalDf.iloc[y_hc==1,1], s=30, c='blue')
#plt.scatter(principalDf.iloc[y_hc==2,0],principalDf.iloc[y_hc==2,1], s=30, c='green')
plt.xlabel('PC1')
plt.ylabel('PC2')

# COMMAND ----------

fp = fp_df.set_index('0')
fp = fp.iloc[:,1:]
reducer = umap.UMAP()
embedding = reducer.fit_transform(fp)
emb_df = pd.DataFrame(data = embedding, index=fp.index, columns=['Embedding_1','Embedding_2'])
hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage ='ward')
y_hc = hc.fit_predict(emb_df)
plt.scatter(emb_df.iloc[y_hc==0, 0], emb_df.iloc[y_hc==0,1],s=20, c='red', label ='Cluster 1')
plt.scatter(emb_df.iloc[y_hc==1, 0], emb_df.iloc[y_hc==1,1], s=20, c='blue', label ='Cluster 2')

plt.xlabel('UMAP_embedding_1')
plt.ylabel('UMAP_embedding_2')

# COMMAND ----------

#hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage ='ward')
#y_hc = hc.fit_predict(principalDf)

ax = plt.axes(projection ="3d")
ax.scatter3D(principalDf.iloc[y_hc==0, 0], principalDf.iloc[y_hc==0,1], principalDf.iloc[y_hc==0,2],s=40, c='red', label ='Cluster 1')
ax.scatter3D(principalDf.iloc[y_hc==1, 0], principalDf.iloc[y_hc==1,1], principalDf.iloc[y_hc==1,2],s=40, c='blue', label ='Cluster 2')
ax.scatter3D(principalDf.iloc[y_hc==2, 0], principalDf.iloc[y_hc==2,1], principalDf.iloc[y_hc==2,2],s=40, c='green', label ='Cluster 3')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# COMMAND ----------

hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage ='ward')
y_hc = hc.fit_predict(principalDf)
plt.scatter(principalDf.iloc[y_hc==0, 0], principalDf.iloc[y_hc==0, 1], s=40, c='red', label ='Cluster 1')
plt.scatter(principalDf.iloc[y_hc==1, 0], principalDf.iloc[y_hc==1, 1], s=40, c='blue', label ='Cluster 2')
#plt.scatter(principalDf.iloc[y_hc==2, 0], principalDf.iloc[y_hc==2, 1], s=20, c='green', label ='Cluster 3')

jitterx = np.random.uniform(low=0, high=0.5, size=len(principalDf))
jittery = np.random.uniform(low=0, high=0.5, size=len(principalDf))
jitx = pd.DataFrame(jitterx)
jity = pd.DataFrame(jittery)
#plt.scatter(jitx.iloc[y_hc==0,0], jity.iloc[y_hc==0,0], s=10, c='pink')
#plt.scatter(jitx.iloc[y_hc==1,0], jity.iloc[y_hc==1,0], s=10, c='cyan') 
#plt.scatter(jitx.iloc[y_hc==2,0], jity.iloc[y_hc==2,0], s=10, c='cyan')
            
plt.xlabel("PC1")
plt.ylabel("PC2")
#plt.scatter(principalDf.iloc[y_hc==3, 0], principalDf.iloc[y_hc==3, 1], s=10, c='black', label ='Cluster 4')
print(silhouette_score(principalDf, y_hc))

# COMMAND ----------

hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage ='ward')
y_hc = hc.fit_predict(principalDf)
#plt.scatter(principalDf.iloc[y_hc==0, 0], principalDf.iloc[y_hc==0, 1], s=40, c='red', label ='Cluster 1')
#plt.scatter(principalDf.iloc[y_hc==1, 0], principalDf.iloc[y_hc==1, 1], s=40, c='blue', label ='Cluster 2')
#plt.scatter(principalDf.iloc[y_hc==2, 0], principalDf.iloc[y_hc==2, 1], s=20, c='green', label ='Cluster 3')

jitterx = np.random.uniform(low=-0.1, high=0.1, size=len(principalDf))
jittery = np.random.uniform(low=-0.1, high=0.1, size=len(principalDf))
jitx = pd.DataFrame(jitterx)
jity = pd.DataFrame(jittery)

plt.scatter(principalDf.iloc[y_hc==0, 0]+jitx.iloc[y_hc==0,0], principalDf.iloc[y_hc==0, 1]+jity.iloc[y_hc==0,0], s=20, c='red', label ='Cluster 1')
plt.scatter(principalDf.iloc[y_hc==1, 0]+jitx.iloc[y_hc==1,0], principalDf.iloc[y_hc==1, 1]+jity.iloc[y_hc==1,0], s=20, c='blue', label ='Cluster 2')
#plt.scatter(jitx.iloc[y_hc==0,0], jity.iloc[y_hc==0,0], s=5, c='pink')
#plt.scatter(jitx.iloc[y_hc==1,0], jity.iloc[y_hc==1,0], s=10, c='cyan') 
#plt.scatter(jitx.iloc[y_hc==2,0], jity.iloc[y_hc==2,0], s=10, c='cyan')
            
plt.xlabel("PC1")
plt.ylabel("PC2")

# COMMAND ----------

jitterx = np.random.uniform(low=-0.1, high=0.1, size=len(principalDf))
jittery = np.random.uniform(low=-0.1, high=0.1, size=len(principalDf))
jitterz = np.random.uniform(low=-0.1, high=0.1, size=len(principalDf))
jitx = pd.DataFrame(jitterx)
jity = pd.DataFrame(jittery)
jitz = pd.DataFrame(jitterz)

ax = plt.axes(projection ="3d")
ax.scatter3D(principalDf.iloc[y_hc==0, 0]+jitx.iloc[y_hc==0,0], principalDf.iloc[y_hc==0,1]+jity.iloc[y_hc==0,0], principalDf.iloc[y_hc==0,2]+jitz.iloc[y_hc==0,0],s=40, c='red', label ='Cluster 1')
ax.scatter3D(principalDf.iloc[y_hc==1, 0]+jitx.iloc[y_hc==1,0], principalDf.iloc[y_hc==1,1]+jity.iloc[y_hc==1,0], principalDf.iloc[y_hc==1,2]+jitz.iloc[y_hc==1,0],s=40, c='blue', label ='Cluster 2')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# COMMAND ----------

#Clusters with PCA on fingerprints
clust0 = pd.DataFrame()
clust1 = pd.DataFrame()
#clust2 = pd.DataFrame()
clust0 = fp_df.iloc[y_hc==0, 0:1]
clust1 = fp_df.iloc[y_hc==1, 0:1]
#clust2 = fp_df.iloc[y_hc==2, 0:1]
clust0.reset_index(inplace=True, drop=True)
clust1.reset_index(inplace=True, drop=True)
#clust2.reset_index(inplace=True, drop=True)
clust0 = clust0.rename(columns = {'0':'patient_id'})
clust1 = clust1.rename(columns = {'0':'patient_id'})
#clust2 = clust2.rename(columns = {'0':'patient_id'})
print(clust0.shape)
print(clust1.shape)
#print(clust2.shape)
#clust2
#spark_df_0=spark.createDataFrame(clust0) 
#spark_df_0.write.parquet("abfss://rdp-phi-sandbox@datalakestorage321.dfs.core.windows.net/arpita_ibd_clust0")
#spark_df_1=spark.createDataFrame(clust1) 
#spark_df_1.write.parquet("abfss://rdp-phi-sandbox@datalakestorage321.dfs.core.windows.net/arpita_ibd_clust1")

# COMMAND ----------

#Clusters with UMAP on fingerprints
clust0 = pd.DataFrame()
clust1 = pd.DataFrame()
#clust2 = pd.DataFrame()
clust0 = fp_df.iloc[y_hc==0, 0:1]
clust1 = fp_df.iloc[y_hc==1, 0:1]
#clust2 = fp_df.iloc[y_hc==2, 0:1]
clust0.reset_index(inplace=True, drop=True)
clust1.reset_index(inplace=True, drop=True)
#clust2.reset_index(inplace=True, drop=True)
clust0 = clust0.rename(columns = {'0':'patient_id'})
clust1 = clust1.rename(columns = {'0':'patient_id'})
#clust2 = clust2.rename(columns = {'0':'patient_id'})
print(clust0.shape)
print(clust1.shape)
#print(clust2.shape)
#clust2
#spark_df_0=spark.createDataFrame(clust0) 
#spark_df_0.write.parquet("abfss://rdp-phi-sandbox@datalakestorage321.dfs.core.windows.net/arpita_ibd_clust0")
#spark_df_1=spark.createDataFrame(clust1) 
#spark_df_1.write.parquet("abfss://rdp-phi-sandbox@datalakestorage321.dfs.core.windows.net/arpita_ibd_clust1")

# COMMAND ----------

#dbutils.fs.rm("abfss://rdp-phi-sandbox@datalakestorage321.dfs.core.windows.net/arpita_ibd_clust0",True)

# COMMAND ----------

diag = spark.sql("SELECT patient_id, crohns_disease, ulcerative_colitis, hepatic_encephalopathy, hepatitis_acute, hepatitis_chronic, cirrhosis_of_liver, biliary_tract_disorder, autoimmune_hepatitis, celiac_disease, chronic_pancreatitis, constipation, cyst, intestinal_obstruction, liver_lesion, liver_cyst, liver_cell_carcinoma, liver_disease_end_stage, liver_enzymes_level_elevated, vitamin_d_deficiency, viral_hepatitis_type_c, viral_hepatitis_type_b, gerd, gerd_without_esophagitis, hyperglycemia, blood_in_urine, blood_leukocyte_number_increased, breast_lump, crohns_disease_of_small_and_large_intestines, crohns_disease_of_ileum, weight_loss, renal_disease_end_stage,renal_transplant_history  FROM rdp_phi_sandbox.hadlock_ml_diagnoses_by_patient")
diag=diag.toPandas()
#diag = diag.set_index('patient_id')
diag.head()

# COMMAND ----------

fp_ibd_clust0 = clust0
fp_ibd_clust1 = clust1
clust0_diag = diag.merge(fp_ibd_clust0, on =['patient_id'],how='inner')
clust0_diag = clust0_diag.set_index('patient_id')
clust1_diag = diag.merge(fp_ibd_clust1, on =['patient_id'],how='inner')
clust1_diag = clust1_diag.set_index('patient_id')
print(clust0_diag.shape)
print(clust1_diag.shape)

# COMMAND ----------

plt.scatter(emb_df.iloc[y_hc==0, 0], emb_df.iloc[y_hc==0,1],s=20, c='red')#, label ='Cluster 1')
plt.scatter(emb_df.iloc[y_hc==1, 0], emb_df.iloc[y_hc==1,1], s=20, c='lightblue')#, label ='Cluster 2')

a = emb_df.merge(clust0_diag,left_index=True,right_index=True, how='inner')
b = emb_df.merge(clust1_diag,left_index=True,right_index=True, how='inner')


#c = a[a.hepatic_encephalopathy==1]
#d = b[b.hepatic_encephalopathy==1]
#plt.scatter(emb_df.loc[c.index, 'Embedding_1'], emb_df.loc[c.index, 'Embedding_2'],s=10, c='black')
#plt.scatter(emb_df.loc[d.index, 'Embedding_1'], emb_df.loc[d.index, 'Embedding_2'],s=10, c='black', label = 'Hepatic Encephalopathy')


#e = a[a.hepatitis_acute==1]
#f = b[b.hepatitis_acute==1]
#plt.scatter(emb_df.loc[e.index, 'Embedding_1'], emb_df.loc[e.index, 'Embedding_2'],s=10, c='green')
#plt.scatter(emb_df.loc[f.index, 'Embedding_1'], emb_df.loc[f.index, 'Embedding_2'],s=10, c='green', label = 'Acute Hepatitis')

g = a[a.renal_disease_end_stage==1]
h = b[b.renal_disease_end_stage==1]
plt.scatter(emb_df.loc[g.index, 'Embedding_1'], emb_df.loc[g.index, 'Embedding_2'],s=10, c='blue')
plt.scatter(emb_df.loc[h.index, 'Embedding_1'], emb_df.loc[h.index, 'Embedding_2'],s=10, c='blue', label = 'End-stage Renal disease')

plt.xlabel('UMAP_embedding_1')
plt.ylabel('UMAP_embedding_2')
plt.legend()

# COMMAND ----------

print(g.shape)
print(h.shape)

# COMMAND ----------

plt.scatter(principalDf.iloc[y_hc==0, 0], principalDf.iloc[y_hc==0,1],s=90, c='red', label ='Cluster 1')
plt.scatter(principalDf.iloc[y_hc==1, 0], principalDf.iloc[y_hc==1,1], s=90, c='blue', label ='Cluster 2')

a = principalDf.merge(clust0_diag,left_index=True,right_index=True, how='inner')
a = a[a.crohns_disease==1]
b = principalDf.merge(clust1_diag,left_index=True,right_index=True, how='inner')
b = b[b.crohns_disease==1]
plt.scatter(principalDf.loc[a.index, 'PC_1'], principalDf.loc[a.index, 'PC_2'],s=90, c='green')
plt.scatter(principalDf.loc[b.index, 'PC_1'], principalDf.loc[b.index, 'PC_2'],s=90, c='green')

x = principalDf.merge(clust0_diag,left_index=True,right_index=True, how='inner')
x = x[x.ulcerative_colitis==1]
y = principalDf.merge(clust1_diag,left_index=True,right_index=True, how='inner')
y = y[y.ulcerative_colitis==1]
plt.scatter(principalDf.loc[x.index, 'PC_1'], principalDf.loc[x.index, 'PC_2'],s=190, c='black')
plt.scatter(principalDf.loc[y.index, 'PC_1'], principalDf.loc[y.index, 'PC_2'],s=190, c='black')

z = principalDf.merge(diag, left_index=True,right_index=True, how='inner')
z = z[z.crohns_disease==1]
z = z[z.ulcerative_colitis==1]
plt.scatter(principalDf.loc[z.index, 'PC_1'], principalDf.loc[z.index, 'PC_2'],s=90, c='yellow')

plt.xlabel('PC1')
plt.ylabel('PC2')

# COMMAND ----------

ax = plt.axes(projection ="3d")
ax.scatter3D(principalDf.iloc[y_hc==0, 0], principalDf.iloc[y_hc==0,1], principalDf.iloc[y_hc==0,2],s=20, c='red', label ='Cluster 1')
ax.scatter3D(principalDf.iloc[y_hc==1, 0], principalDf.iloc[y_hc==1,1], principalDf.iloc[y_hc==1,2],s=20, c='blue', label ='Cluster 2')

a = principalDf.merge(clust0_diag,left_index=True,right_index=True, how='inner')
a = a[a.crohns_disease==1]
b = principalDf.merge(clust1_diag,left_index=True,right_index=True, how='inner')
b = b[b.crohns_disease==1]
ax.scatter3D(principalDf.loc[a.index, 'PC_1'], principalDf.loc[a.index, 'PC_2'], principalDf.loc[a.index, 'PC_3'],s=90, c='green')
ax.scatter3D(principalDf.loc[b.index, 'PC_1'], principalDf.loc[b.index, 'PC_2'], principalDf.loc[b.index, 'PC_3'],s=90, c='green')

x = principalDf.merge(clust0_diag,left_index=True,right_index=True, how='inner')
x = x[x.ulcerative_colitis==1]
y = principalDf.merge(clust1_diag,left_index=True,right_index=True, how='inner')
y = y[y.ulcerative_colitis==1]
ax.scatter3D(principalDf.loc[x.index, 'PC_1'], principalDf.loc[x.index, 'PC_2'], principalDf.loc[x.index, 'PC_3'],s=210, c='black')
ax.scatter3D(principalDf.loc[y.index, 'PC_1'], principalDf.loc[y.index, 'PC_2'], principalDf.loc[y.index, 'PC_3'],s=210, c='black')

diag = diag.set_index('patient_id')

z = principalDf.merge(diag, left_index=True,right_index=True, how='inner')
z = z[z.crohns_disease==1]
z = z[z.ulcerative_colitis==1]
ax.scatter3D(principalDf.loc[z.index, 'PC_1'], principalDf.loc[z.index, 'PC_2'], principalDf.loc[z.index, 'PC_3'], s=150, c='yellow')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# COMMAND ----------

fp_ibd_clust0 = fp_ibd_clust0.set_index('patient_id')
fp_ibd_clust1 = fp_ibd_clust1.set_index('patient_id')
ibd_labs_clust0 = ibd_pandas.merge(fp_ibd_clust0,left_index=True, right_index=True,how='inner')
ibd_labs_clust1 = ibd_pandas.merge(fp_ibd_clust1,left_index=True,right_index=True,how='inner')

# COMMAND ----------

c0 = ibd_labs_clust0.notnull().sum()
c1 = ibd_labs_clust1.notnull().sum()
plt.hist(c0, bins='auto')
plt.hist(c1, bins='auto')

# COMMAND ----------

for i in c0.index:
  if 'ibd' in i or 'first' in i or 'last' in i or 'neg' in i or 'pos' in i:
    c0 = c0.drop(labels=i)
    c1 = c1.drop(labels=i)

# COMMAND ----------

ibd_labs_clust0_red = ibd_labs_clust0
ibd_labs_clust1_red = ibd_labs_clust1
for col in ibd_labs_clust0:
    if col not in c0:
        ibd_labs_clust0_red = ibd_labs_clust0_red.drop([col], axis=1)
        ibd_labs_clust1_red = ibd_labs_clust1_red.drop([col], axis=1)

# COMMAND ----------

a = ibd_labs_clust0_red.iloc[:,1:]
b = ibd_labs_clust1_red.iloc[:,1:]
a1_hi = pd.Series(index= a.columns)
a1_lo = pd.Series(index= a.columns)
b1_hi = pd.Series(index= b.columns)
b1_lo = pd.Series(index= b.columns)
for col in a:
  a1_hi.loc[col] = len(a[a[col] == 1])
  b1_hi.loc[col] = len(b[b[col] == 1])
  a1_lo.loc[col] = len(a[a[col] == 0])
  b1_lo.loc[col] = len(b[b[col] == 0])

# COMMAND ----------

from scipy.stats import chi2_contingency
ibd_chi = pd.DataFrame({"lab": [], "chi_stat": [], "pval": [], "dof": [], "expctd": [], "observed": []})
index = ['high','low']
columns = ['clust0','clust1']
i=0
for row in a:
  ibd_chi.loc[i,'lab'] = row
  contingency = pd.DataFrame(data=[[a1_hi[row],b1_hi[row]],[a1_lo[row],b1_lo[row]]], index=index, columns=columns)
  ibd_chi.loc[i,'observed'] = str(a1_hi[row])+', '+str(b1_hi[row])+', '+str(a1_lo[row])+', '+str(b1_lo[row])
  try:
    c,p,d,e = chi2_contingency(contingency)
  except Exception as ex:
    pass
  #print(contingency)
  ibd_chi.loc[i,'chi_stat'] = c
  ibd_chi.loc[i,'pval'] = p
  ibd_chi.loc[i,'dof'] = d
  e=e.flatten()
  #e_arr = []
  #e_arr = e_arr.append(e[0])
  #e_arr = e_arr.append(e[1])
  #e_arr = e_arr.append(e[2])
  #e_arr = e_arr.append(e[3])
  ibd_chi.loc[i,'expctd'] = str(round(e[0],2))+', '+str(round(e[1],2))+', '+str(round(e[2],2))+', '+str(round(e[3],2))
  i = i+1

# COMMAND ----------

ibd_chi = ibd_chi.sort_values(by='chi_stat', ascending=False, ignore_index=True)

# COMMAND ----------

ibd_chi

# COMMAND ----------

dbutils.fs.rm("abfss://rdp-phi-sandbox@datalakestorage321.dfs.core.windows.net/arpita_ibd_chi",True)
spark_df=spark.createDataFrame(ibd_chi) 
spark_df.write.parquet("abfss://rdp-phi-sandbox@datalakestorage321.dfs.core.windows.net/arpita_ibd_chi")

# COMMAND ----------

contingency = pd.DataFrame(data=[[a1_hi[0],b1_hi[0]],[a1_lo[0],b1_lo[0]]], index=index, columns=columns)
contingency

# COMMAND ----------

c, p, d, e = chi2_contingency(contingency)
chi2_contingency(contingency)

# COMMAND ----------

chi_df = spark.read.parquet("abfss://rdp-phi-sandbox@datalakestorage321.dfs.core.windows.net/arpita_ibd_chi")
chi_df=chi_df.toPandas()

# COMMAND ----------

chi_df = chi_df.sort_values(by='chi_stat', ascending=False, ignore_index=True)

# COMMAND ----------

obs = chi_df.loc[:,'observed']
exp = chi_df.loc[:,'expctd']

# COMMAND ----------

obs=obs.to_frame()
exp = exp.to_frame()

# COMMAND ----------

obs = pd.DataFrame(obs.observed.str.split(', ',3).tolist(),columns=['a','b','c','d'])
obs = obs.astype(float)

# COMMAND ----------

exp = pd.DataFrame(exp.expctd.str.split(', ',3).tolist(),columns=['a','b','c','d'])
exp = exp.astype(float)

# COMMAND ----------

for i in range(len(obs)):
  print(str(obs.loc[i,'a']/exp.loc[i,'a']) +' '+ str(obs.loc[i,'b']/exp.loc[i,'b']) +' '+ str(obs.loc[i,'c']/exp.loc[i,'c']) + ' ' +str(obs.loc[i,'d']/exp.loc[i,'d']))

# COMMAND ----------



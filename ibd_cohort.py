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

// Databricks notebook source
// MAGIC %md ### 3252 - Course Project: Telco Customer Churn Prediction
// MAGIC 
// MAGIC Author: Shaikh Asif 

// COMMAND ----------

// MAGIC %md ### 1. Data Exploration and Preparation
// MAGIC 
// MAGIC #### Content:
// MAGIC    The IBM Sample Dataset has information about Telco customers and if they left the company within the last month (churn). Each observation represents a unique costumer, the columns contains information about customer’s services, account and socio data
// MAGIC 
// MAGIC ##### The data set includes information about:
// MAGIC   - Customers who left within the last month – the column is called Churn
// MAGIC   - Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, 
// MAGIC     device protection, tech support, and streaming TV and movies
// MAGIC   - Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, 
// MAGIC     monthly charges, and total charges
// MAGIC   - Demographic info about customers – gender, age seniority and if they have partners and dependents

// COMMAND ----------

// Import the data:
val data = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv("/FileStore/tables/WA_Fn_UseC__Telco_Customer_Churn-89c80.csv")

// Preview imported data:
display(data)

// COMMAND ----------

// The TotalCharges attribute is recognized as a String data type but from the data head preview above, the values
// are floating point numeric so there has to be some values that makes spark transform it into the String data type.

// Check data for any null values:
import org.apache.spark.sql.functions.{sum, col}
data.select(data.columns.map(c => sum(col(c).isNull.cast("int")).alias(c)): _*).show


// COMMAND ----------

// No nulls found above

// Check data for any blank values since some of our imported columns are of the String type:
data.select(data.columns.map(c => sum(col(c).equalTo(" ").cast("int")).alias(c)): _*).show

// COMMAND ----------

// There are 11 observations where TotalCharges is blank.
// Let's look at these observations:
display(data.filter("TotalCharges == ' '"))

// COMMAND ----------

// The TotalCharges are null for 11 customers with 0 months tenure that didn't churn but their monthly charges are available.
// Since there are only 11 obs for 0 months tenure customers, we can rely on training our classifiers on the other low tenure
// customers for predicting churn when tenure is low. 

// We can therefore go ahead and remove these 11 obs:
val data_v2 = data.filter("TotalCharges != ' '").toDF()

// COMMAND ----------

data_v2.filter("TotalCharges = ' '").count()

// COMMAND ----------

// 1. We now need to convert TotalCharges from String to Double
// 2. We also need to convert SeniorCitizen from integer to String type as it is a categorical variable
// 3. We also need to convert our target column Churn from String to a numeric data type
// 4. Let us also drop the customerID column as it is no use to the model in predicting churn

// Convert the dataframe to a SQL view to perform these transformations:
data_v2.createOrReplaceTempView("data_v2")

// COMMAND ----------

sqlContext.cacheTable("data_v2")

// Convert to a dataset from the table - make sure we only keep the relevant fields and data:
val data_v3 = spark.sql("""select gender as Gender, 
                                case when SeniorCitizen = 0 then "No" else "Yes" end as SeniorCitizen, 
                                Partner, Dependents, tenure as Tenure, PhoneService, MultipleLines,
                                InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
                                TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling,
                                PaymentMethod, MonthlyCharges, 
                                cast(TotalCharges as double) as TotalCharges,
                                case Churn when "No" then 0 else 1 end as Churn
                         from data_v2""")

// COMMAND ----------

// Let's look at the stats summary of the continous variables/features
display(data_v3.describe())

// COMMAND ----------

// Lets explore the features through visualization:
display(data_v3.select("Tenure", "MonthlyCharges", "TotalCharges", "Churn"))

// COMMAND ----------

display(data_v3.select("Tenure", "TotalCharges"))

// COMMAND ----------

// We can see that TotalCharges is highly correlated with tenure, this will cause collinearity and potentially an unstable
//intercept for our models. It makes sense that the higher the tenure, the more the customer has paid over that duration.
// We can go ahead and remove this attribute from the data:

val data_v4 = data_v3.drop("TotalCharges")

// COMMAND ----------

// Analyze the categorical features:
display(data_v4)

// COMMAND ----------

display(data_v4)

// COMMAND ----------

display(data_v4)

// COMMAND ----------

display(data_v4)

// COMMAND ----------

// High proportion of customers have no dependents, are non-senior citizens, have phone service, are not on contracts, don't
// have device protection, tech support and online security

// COMMAND ----------

display(data_v4)

// COMMAND ----------

// Senior Citizens and customers with no dependents and no partner tend to churn more.

// COMMAND ----------

display(data_v4)

// COMMAND ----------

// Customers who have Fibre Optic, no online security, backup, device protection or tech support tend to churn more.
// Customers who don't have internet service tend to be not sticky and that is possible a cause to churn more

// COMMAND ----------

display(data_v4)

// COMMAND ----------

// Customers on no contract, have paperless billing and pay by electronic check tend to churn more.

// COMMAND ----------

data_v4.createOrReplaceTempView("churners_by_tenure")

// COMMAND ----------

display(spark.sql("select tenure, count(*) as Num_Churners from churners_by_tenure where Churn = 1 group by tenure order by tenure"))

// COMMAND ----------

// We see a trend of lower the tenure the more the churners

// COMMAND ----------

display(data_v4.filter("Churn == 1"))

// COMMAND ----------

// Customers with a higher monthly bills tend to churn more.

// COMMAND ----------

// MAGIC %md ### 2. Data Pre-Processing

// COMMAND ----------

// Now that all of our data is prepped. We're going to have to put all of it into one column of a vector type for Spark MLLib. 
// This makes it easy to embed a prediction right in a DataFrame and also makes it very clear as to what is getting passed into the model and what isn't

// First we need to 1-hot encode the categorical variables:
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, OneHotEncoder}
                                
 
val gen_ind = new StringIndexer().setInputCol("Gender").setOutputCol("genInd")
val sen_ind = new StringIndexer().setInputCol("SeniorCitizen").setOutputCol("senInd")
val par_ind = new StringIndexer().setInputCol("Partner").setOutputCol("parInd")
val dep_ind = new StringIndexer().setInputCol("Dependents").setOutputCol("depInd")
val phn_ind = new StringIndexer().setInputCol("PhoneService").setOutputCol("phnInd")
val mul_ind = new StringIndexer().setInputCol("MultipleLines").setOutputCol("mulInd")
val int_ind = new StringIndexer().setInputCol("InternetService").setOutputCol("intInd")
val onl_ind = new StringIndexer().setInputCol("OnlineSecurity").setOutputCol("onlInd")
val onb_ind = new StringIndexer().setInputCol("OnlineBackup").setOutputCol("onbInd")
val dev_ind = new StringIndexer().setInputCol("DeviceProtection").setOutputCol("devInd")
val tec_ind = new StringIndexer().setInputCol("TechSupport").setOutputCol("tecInd")
val str_ind = new StringIndexer().setInputCol("StreamingTV").setOutputCol("strInd")
val con_ind = new StringIndexer().setInputCol("Contract").setOutputCol("conInd")
val pap_ind = new StringIndexer().setInputCol("PaperlessBilling").setOutputCol("papInd")
val pay_ind = new StringIndexer().setInputCol("PaymentMethod").setOutputCol("payInd")

val cat_indexers = Array(gen_ind, sen_ind, par_ind, dep_ind, phn_ind, mul_ind, int_ind, onl_ind, onb_ind, dev_ind, tec_ind, str_ind, con_ind, pap_ind, pay_ind)

val gen_end = new OneHotEncoder().setInputCol("genInd").setOutputCol("genVec")
val sen_end = new OneHotEncoder().setInputCol("senInd").setOutputCol("senVec")
val par_end = new OneHotEncoder().setInputCol("parInd").setOutputCol("parVec")
val dep_end = new OneHotEncoder().setInputCol("depInd").setOutputCol("depVec")
val phn_end = new OneHotEncoder().setInputCol("phnInd").setOutputCol("phnVec")
val mul_end = new OneHotEncoder().setInputCol("mulInd").setOutputCol("mulVec")
val int_end = new OneHotEncoder().setInputCol("intInd").setOutputCol("intVec")
val onl_end = new OneHotEncoder().setInputCol("onlInd").setOutputCol("onlVec")
val onb_end = new OneHotEncoder().setInputCol("onbInd").setOutputCol("onbVec")
val dev_end = new OneHotEncoder().setInputCol("devInd").setOutputCol("devVec")
val tec_end = new OneHotEncoder().setInputCol("tecInd").setOutputCol("tecVec")
val str_end = new OneHotEncoder().setInputCol("strInd").setOutputCol("strVec")
val con_end = new OneHotEncoder().setInputCol("conInd").setOutputCol("conVec")
val pap_end = new OneHotEncoder().setInputCol("papInd").setOutputCol("papVec")
val pay_end = new OneHotEncoder().setInputCol("payInd").setOutputCol("payVec")

val cat_encoders = Array(gen_end, sen_end, par_end, dep_end, phn_end, mul_end, int_end, onl_end, onb_end, dev_end, tec_end, str_end, con_end, pap_end, pay_end)

// Put all features in one vector column
val assembler = new VectorAssembler()
  .setInputCols(Array("genVec","senVec","parVec","depVec","phnVec","mulVec","intVec","onlVec","onbVec","devVec","tecVec","strVec","conVec","papVec","payVec", 
                      "Tenure", "MonthlyCharges"))
  .setOutputCol("features")

// COMMAND ----------

// Build the Logistic Regression model and parameter grid:
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}

val lr = new LogisticRegression()
  .setLabelCol("Churn")
  .setFeaturesCol("features")

val paramGrid = new ParamGridBuilder()
  .addGrid(lr.maxIter, Array(10, 20))
  .addGrid(lr.regParam, Array(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
  .addGrid(lr.elasticNetParam, Array(0.0, 0.4, 0.8, 1.0))
  .build()

// COMMAND ----------

// Build the pipeline:
import org.apache.spark.ml.{Pipeline, PipelineStage}
val steps: Array[PipelineStage] = cat_indexers ++ cat_encoders ++ Array(assembler, lr)
val pipeline = new Pipeline().setStages(steps)

// Add cross-validation:
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
val cv = new CrossValidator() 
  .setEstimator(pipeline) 
  .setEstimatorParamMaps(paramGrid)
  .setEvaluator(new BinaryClassificationEvaluator().setLabelCol("Churn"))

//Split into training and test sets:
val splitSeed = 5043
val Array(training, test) = data_v4.randomSplit(Array(0.7, 0.3), splitSeed)

training.cache()
test.cache()

println(training.count()) 
println(test.count())

// COMMAND ----------

// Fit and train the model:
val fitted_lr = cv.fit(training)

// COMMAND ----------

// Get the best fit params from the cv folds:
println("The Best Parameters:\n--------------------")
println(fitted_lr.bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel].stages(31))
fitted_lr
  .bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel]
  .stages(31)
  .extractParamMap

// COMMAND ----------

// Run against the test set to see how the model performs
val lr_predictions = fitted_lr.bestModel
  .transform(test)
  .selectExpr("rawPrediction", "prediction", "probability", "Churn")
display(lr_predictions)

// COMMAND ----------

// Model evaluation metrics:
//A common metric used for logistic regression is area under the ROC curve (AUC) (max area is 1):
val evaluator = new BinaryClassificationEvaluator().setLabelCol("Churn").setRawPredictionCol("rawPrediction").setMetricName("areaUnderROC")

// Evaluates predictions and returns the AUC (larger is better)
val accuracy = evaluator.evaluate(lr_predictions)

// COMMAND ----------

// Other Model Evaluation Metrics:
val cp = lr_predictions.select( "Churn", "prediction")
val counttotal = lr_predictions.count()
val correct = cp.filter($"Churn" === $"prediction").count()
val wrong = cp.filter(!($"Churn" === $"prediction")).count()
val truep = cp.filter($"prediction" === 0.0).filter($"Churn" === $"prediction").count()
val falseN = cp.filter($"prediction" === 0.0).filter(!($"Churn" === $"prediction")).count()
val falseP = cp.filter($"prediction" === 1.0).filter(!($"Churn" === $"prediction")).count()
val ratioWrong=wrong.toDouble/counttotal.toDouble
val ratioCorrect=correct.toDouble/counttotal.toDouble

// COMMAND ----------

// Build the Random Forest Classifier model and parameter grid:
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}

val rf = new RandomForestClassifier()
  .setLabelCol("Churn")
  .setFeaturesCol("features")

// Hyper-Tuning Param Grid:
val paramGrid2 = new ParamGridBuilder()
  .addGrid(rf.maxDepth, Array(2, 4, 6))
  .addGrid(rf.maxBins, Array(20, 40, 60))
  .addGrid(rf.numTrees, Array(5, 10, 15, 20))
  .build()

val steps2: Array[PipelineStage] = cat_indexers ++ cat_encoders ++ Array(assembler, rf)
val pipeline2 = new Pipeline().setStages(steps2)

// cross-validation:
val cv2 = new CrossValidator() 
  .setEstimator(pipeline2) 
  .setEstimatorParamMaps(paramGrid2)
  .setEvaluator(new BinaryClassificationEvaluator().setLabelCol("Churn"))

// COMMAND ----------

// Fit and train the model:
val fitted_rf = cv2.fit(training)

// COMMAND ----------

// Get the best fit params from the cv folds:
println("The Best Parameters:\n--------------------")
println(fitted_rf.bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel].stages(31))
fitted_rf
  .bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel]
  .stages(31)
  .extractParamMap

// COMMAND ----------

// Run against the test set to see how the model performs
val rf_predictions = fitted_rf.bestModel
  .transform(test)
  .selectExpr("features","rawPrediction", "prediction", "probability", "Churn")
display(rf_predictions)

// COMMAND ----------

// Model evaluation metrics:
// AUC:
val rf_evaluator = new BinaryClassificationEvaluator().setLabelCol("Churn").setRawPredictionCol("rawPrediction").setMetricName("areaUnderROC")
val rf_accuracy = rf_evaluator.evaluate(rf_predictions)

// COMMAND ----------

// Features importance with Random Forest Classifier:

import org.apache.spark.ml.attribute._
import org.apache.spark.sql.functions._

val bestModel = fitted_rf.bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel]
val lstModel = bestModel.stages.last.asInstanceOf[RandomForestClassificationModel]
val ftrImp = lstModel.featureImportances.toArray
val schema = rf_predictions.schema

val featureAttrs = AttributeGroup.fromStructField(schema(lstModel.getFeaturesCol)).attributes.get
val mfeatures = featureAttrs.map(_.name.get)


val mdf = sc.parallelize(mfeatures zip ftrImp).toDF("featureName","Importance").orderBy(desc("Importance"))
display(mdf)

// COMMAND ----------



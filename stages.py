from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.sql.functions import col, udf
from pyspark.ml.functions import vector_to_array
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import classification_report, confusion_matrix
from pyspark.ml.classification import LogisticRegression, OneVsRest
from pyspark.sql.types import FloatType, ArrayType, DoubleType
import glob
import numpy as np


stages = []
spark = (SparkSession
        .builder
        .appName("Clustering model")
        .getOrCreate())

filename = glob.glob("preprocessed_data/*.csv")
#filename = "train_data_clean.csv"

df = (spark.read.format("csv")
        .option("header", "true")
        .option("delimiter", "\t")
        .load(filename))
# df.show(n=5, truncate=False)
# print(df.count(), len(df.columns))

id_cols = ['code','product_name']
nutrition_cols = ['energy_100g',
                    'proteins_100g',
                    'fat_100g',
                    'carbohydrates_100g',
                    'sugars_100g',
                    'energy-kcal_100g',
                    'saturated-fat_100g',
                    'salt_100g',
                    'sodium_100g',
                    'fiber_100g',
                    'fruits-vegetables-nuts-estimate-from-ingredients_100g',
                    'nutrition-score-fr_100g']

df = df[id_cols + nutrition_cols]
df = df.dropna()
# print("SIZE DATA", dataset.count())
feat_df = df#[nutrition_cols]
# feat_df.printSchema()

feat_df = feat_df.select(*(col(c).cast("float").alias(c) for c in feat_df.columns))
# feat_df.printSchema()

assemble = VectorAssembler(inputCols=nutrition_cols, outputCol='features')
# assembled_data = assemble.transform(feat_df) 1
# assembled_data.show(2)

scale = StandardScaler(inputCol='features', outputCol='standardized')
# data_scale = scale.fit(assembled_data) 2
# data_scale_output = data_scale.transform(assembled_data) 3
# data_scale_output.show(2)

# Trains a k-means model.
KMeans_algo = KMeans(featuresCol='standardized', predictionCol='clust_preds', k=7)
# KMeans_fit = KMeans_algo.fit(data_scale_output) 4

# Make predictions
# output = KMeans_fit.transform(data_scale_output) 5

# CUSTOM TRANSFORMER ----------------------------------------------------------------
class ColumnDropper(Transformer):
    """
    A custom Transformer which drops all columns that have at least one of the
    words from the banned_list in the name.
    """

    def __init__(self, banned_list):
        super(ColumnDropper, self).__init__()
        self.banned_list = banned_list

    def _transform(self, df):
        df = df.drop(*[x for x in df.columns if any(y in x for y in self.banned_list)])
        return df


column_dropper = ColumnDropper(banned_list = ["rawPrediction", "probability"])

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="clust_preds", featuresCol="standardized", predictionCol='class_preds', numTrees=10)

# Third model LR
# lin_reg = LinearRegression(featuresCol='standardized', labelCol='clust_preds') 
lr = LogisticRegression(maxIter=10, tol=1E-6, fitIntercept=True, labelCol="clust_preds", featuresCol="standardized", probabilityCol="lr_prob")
# ovr = OneVsRest(classifier=lr, labelCol="clust_preds", featuresCol="standardized")

# Configure an ML pipeline, which consists of our stages
pipeline = Pipeline(stages=[assemble, scale, KMeans_algo, rf, column_dropper, lr])

# Fit the pipeline to training documents.
model = pipeline.fit(feat_df)
output = model.transform(feat_df)

# Evaluate clustering by computing Silhouette score
clust_evaluator = ClusteringEvaluator(predictionCol='clust_preds', featuresCol='standardized', \
                                metricName='silhouette', distanceMeasure='squaredEuclidean')

# Select (prediction, true label) and compute test error
class_evaluator = MulticlassClassificationEvaluator(labelCol="clust_preds", predictionCol="class_preds", metricName="accuracy")
accuracy = class_evaluator.evaluate(output)
y_true = np.array(output.select("clust_preds").collect())
y_pred = np.array(output.select("class_preds").collect())

score = clust_evaluator.evaluate(output)
print("Clustering Report: silhouette with squared euclidean distance = " + str(score))
print()
print("Classification (1 - accuracy): error = %g" % (1.0 - accuracy))
print("Classification Report:")
print(classification_report(y_true, y_pred, zero_division=1))
print("Classification Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
output = output.withColumn("class_preds", col("class_preds").cast("integer"))
output = output.withColumn("lr_prob", vector_to_array("lr_prob"))
get_proba = udf(lambda x, y: x[y], returnType=DoubleType())
output = output.withColumn('lr_estim_rf', get_proba(output['lr_prob'], output['class_preds']))
print()
print("Logistic Regression Estimation of RF labels:")

# output.printSchema()
# print(np.array(output.select("lr_estim_rf").collect()).squeeze())

output.select(["clust_preds", "class_preds", "lr_estim_rf"]).show(n=5)
# Write CSV file with column header (column names)
# result = output.select(id_cols + nutrition_cols + ["clust_preds", "class_preds", "lr_estim_rf"])
# result.write.option("header",True).option("delimiter", "\t").csv("clustering_data")







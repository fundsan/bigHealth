package jake.mortality.base
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SQLContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import org.postgresql
import org.apache.spark.sql.functions._
/**
  * Created by jake on 4/25/16.
  */
object Base {
  def run(sqlContext: SQLContext){

  val sapsii = sqlContext.load("jdbc", Map("driver" -> "org.postgresql.Driver", "url" -> "jdbc:postgresql://mimic3-1.coh8b3jmul4y.us-west-2.rds.amazonaws.com:5432/MIMIC?user=jfund&password=G0tbeared", "dbtable" -> "sapsiiicu"))

  val pat = sqlContext.load("jdbc", Map("url" -> "jdbc:postgresql://mimic3-1.coh8b3jmul4y.us-west-2.rds.amazonaws.com:5432/MIMIC?user=jfund&password=G0tbeared", "dbtable" -> "mimiciii.patients"))

  val joinPatSap = sapsii.join(pat, "subject_id")

  val values = joinPatSap.select(joinPatSap("subject_id"), joinPatSap("gender"), (datediff(joinPatSap("intime"), joinPatSap("dob")) / 365).as("age").cast("double"), joinPatSap("sapsii"), joinPatSap("dod"), joinPatSap("outtime"))

  val firstVal2 = values.groupBy("subject_id").min("age")

  val afterDrop = firstVal2.join(values, "subject_id")
  val afterFilter = afterDrop.filter(afterDrop("min(age)") === afterDrop("age"))

  val features = afterFilter.drop("min(age)")
  val filteredAge = features.filter(features("age") > 18)
  val toStringToOne = udf((t: String) => (if (t == null) 0 else 1))
  val genderToValue = udf((t: String) => (if (t.equals("M")) 0 else 1))

  val featDF = filteredAge.withColumn("dod", toStringToOne(features("dod"))).withColumn("gender", genderToValue(features("gender")))
  val notDead = featDF.filter(featDF("dod") === 0)
  val dead = featDF.filter(featDF("dod") === 1)

  val feats = featDF.drop("subject_id").rdd
  val vectored = feats.map(row => LabeledPoint(row.getInt(3), Vectors.dense(row.getInt(0), row.getDouble(1).toInt, row.getInt(2))))

  // Split data into training (60%) and test (40%).
  val splits = vectored.randomSplit(Array(0.7, 0.3), seed = 11L)
  val training = splits(0).cache()
  val test = splits(1)

  // Run training algorithm to build the model
  val numIterations = 100
  val model = SVMWithSGD.train(training, numIterations)

  // Clear the default threshold.
  model.clearThreshold()

  // Compute raw scores on the test set.
  val scoreAndLabels = test.map { point =>
    val score = model.predict(point.features)
    (score, point.label)
  }

  // Get evaluation metrics.
  val metrics = new BinaryClassificationMetrics(scoreAndLabels)
  val auROC = metrics.areaUnderROC()

  println("Area under ROC = " + auROC)
}
}

package jake.mortality.combined

import jake.mortality.tdf.TdfText
import jake.mortality.topics.Topics
import model.hadmAndFeatures
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.{DenseVector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SQLContext}

/**
  * Created by jake on 4/25/16.
  */
object Combined {
  def run(sqlContext: SQLContext, hours: String, numTopics: Int) {
    val sc = sqlContext.sparkContext
    val icu = sqlContext.load("jdbc", Map("driver" -> "org.postgresql.Driver", "url" -> "jdbc:postgresql://mimic3-1.coh8b3jmul4y.us-west-2.rds.amazonaws.com:5432/MIMIC?user=jfund&password=*", "dbtable" -> "MIMICIII.ICUSTAYS")).cache()

    val notes = sqlContext.load("jdbc", Map("driver" -> "org.postgresql.Driver", "url" -> "jdbc:postgresql://mimic3-1.coh8b3jmul4y.us-west-2.rds.amazonaws.com:5432/MIMIC?user=jfund&password=*", "dbtable" -> "MIMICIII.NOTEEVENTS")).cache()

    val minICU: DataFrame = icu.groupBy("subject_id").agg(min("intime")).toDF("subject_id", "mintime").cache()

    val joinICUBack: DataFrame = minICU.join(icu, "subject_id").drop("subject_id").cache()

    val filteredBack = joinICUBack.filter(joinICUBack("mintime") === joinICUBack("intime")).cache()

    val noteJoin = notes.join(filteredBack, "hadm_id").drop("row_id").cache()

    val filteredNotes = noteJoin.filter((noteJoin("intime") + expr("INTERVAL "+hours+ " HOUR") >
      noteJoin("charttime")) && (noteJoin("intime") + expr("INTERVAL "+hours+ " HOUR") < noteJoin("outtime")))
    val features = Topics.run(sqlContext, filteredNotes, numTopics, 500)

    val adm = sqlContext.load("jdbc", Map("driver" -> "org.postgresql.Driver", "url" -> "jdbc:postgresql://mimic3-1.coh8b3jmul4y.us-west-2.rds.amazonaws.com:5432/MIMIC?user=jfund&password=*",
      "dbtable" -> "mimiciii.admissions")).cache()

    // get target function
    val toStringToOne = udf((t: String) => if (t == null) 0.0 else 1.0)
    val genderToValue = udf((t: String) => if (t.equals("M")) 0.0 else 1.0)

    val admDF = adm.withColumn("deathtime", toStringToOne(adm("deathtime")))
    val notDead = admDF.filter(admDF("deathtime") === 0).sample(false, .5)
    val dead = admDF.filter(admDF("deathtime") === 1)
    val reissuedDF = notDead.unionAll(dead)

    val joinedAdmFeatures = features.join(reissuedDF, "hadm_id")

    val sapsii = sqlContext.load("jdbc", Map("driver" -> "org.postgresql.Driver", "url" -> "jdbc:postgresql://mimic3-1.coh8b3jmul4y.us-west-2.rds.amazonaws.com:5432/MIMIC?user=jfund&password=*", "dbtable" -> "sapsiiicu"))

    val pat = sqlContext.load("jdbc", Map("url" -> "jdbc:postgresql://mimic3-1.coh8b3jmul4y.us-west-2.rds.amazonaws.com:5432/MIMIC?user=jfund&password=*", "dbtable" -> "mimiciii.patients"))

    val joinPatSap = sapsii.join(pat, "subject_id")

    val values = joinPatSap.select(joinPatSap("subject_id"), joinPatSap("gender"), (datediff(joinPatSap("intime"), joinPatSap("dob")) / 365).as("age").cast("double"), joinPatSap("sapsii"), joinPatSap("dod"), joinPatSap("outtime"))

    val firstVal2 = values.groupBy("subject_id").min("age")

    val afterDrop = firstVal2.join(values, "subject_id")
    val afterFilter = afterDrop.filter(afterDrop("min(age)") === afterDrop("age"))

    val newfeatures = afterFilter.drop("min(age)")
    val filteredAge = features.filter(features("age") > 18)


    val featDF = filteredAge.withColumn("dod", toStringToOne(features("dod"))).withColumn("gender", genderToValue(features("gender")))
   val vectored = featDF.join(joinedAdmFeatures,"subject_id").select("deathtime", "gender","age","sapsii" ,"features").map(row =>LabeledPoint(row.getDouble(0), Vectors.dense(row.getSeq(4).toArray[Double].union(Array(row.getInt(1),row.getDouble
   (2),row.getDouble(3))))))



    // Split data into training (60%) and test (40%).
    vectored.saveAsObjectFile("s3n://jakemimc/basevectored"+hours)
    val splits = vectored.randomSplit(Array(0.7, 0.3), seed = 11L)
    val training = splits(0).cache()
    val testing = splits(1)
    // Run training algorithm to build the model
    val numIterations = 500

    val model = SVMWithSGD.train(training, numIterations,)
    // Clear the default threshold.
    model.clearThreshold()
    // Compute raw scores on the test set.
    val scoreAndLabels = testing.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()
    val output = sc.parallelize(Seq(auROC, String.valueOf(dead.count()), String.valueOf(notDead.count())))
    output.saveAsTextFile("s3n://jakemimc/Combinedoutput"+hours)
    println("Area under ROC = " + auROC)
    println("dead = " + dead.count())
    println("alive = " + notDead.count())
  }
}

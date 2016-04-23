package jake.mortality.main


import java.sql.Timestamp


import jake.mortality.jake.mortality.topics.Topics
import model.Note
import jake.mortality.lin.{toBreezeVector, toBreezeMatrix}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, Tokenizer}
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.clustering.{LDA, DistributedLDAModel}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.{DenseVector, Vectors, SparseVector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.postgresql
import org.apache.spark.sql.functions._


/**
  * Created by jake on 4/14/16.
  */
object Main {


  def main(args: Array[String]) {

    val sc = createContext("final", "local")
   val sqlContext = new SQLContext(sc)
   val testDF = sc.textFile("/Users/jake/dataFinal/output.txt").map(row => row.split('|').map(s => s.replace("  ", ""))).collect
    testDF(0) = testDF(0).map(s => s.replace(" ", ""))
    var finishedArray = List[Array[String]]()
    var cur = 0
    var arrcur = 0

    var i = 0
    for (i <- testDF.indices) {
      val current = testDF(i)

      if (current.last.size > 0 && current.last.charAt(current.last.size - 1).equals('+')) {

        testDF(cur).update(testDF(cur).length - 1, testDF(cur).last.concat(current.last.replace("+", " ")))
      }
      else {

        finishedArray = finishedArray ::: List(testDF(cur))
        cur = i + 1

      }
    }
    val dfs = finishedArray.toSeq.slice(2, finishedArray.size - 3).map(s =>
      (s(0).replace(" ", "").toInt, s(1).replace(" ", "").toInt, Timestamp.valueOf(s(2).stripPrefix(" ").stripSuffix(" ")), Timestamp.valueOf(s(3).stripPrefix(" ").stripSuffix(" ")),
        s(5), Timestamp.valueOf(s(6).stripPrefix(" ").stripSuffix(" ")), s(7)))


    val icu = sqlContext.load("jdbc", Map("driver" -> "org.postgresql.Driver", "url" -> "jdbc:postgresql://mimic3-1.coh8b3jmul4y.us-west-2.rds.amazonaws.com:5432/MIMIC?user=jfund&password=G0tbeared", "dbtable" -> "MIMICIII.ICUSTAYS"))

    val notes = sqlContext.load("jdbc", Map("driver" -> "org.postgresql.Driver", "url" -> "jdbc:postgresql://mimic3-1.coh8b3jmul4y.us-west-2.rds.amazonaws.com:5432/MIMIC?user=jfund&password=G0tbeared", "dbtable" -> "MIMICIII.NOTEEVENTS"))

    val test = sqlContext.createDataFrame(finishedArray.slice(2, finishedArray.size - 3).map(f = s =>
      Note(subject_id = s(0).replace(" ", "").toInt, hadm_id = s(1).replace(" ", "").toInt, chartdate = Timestamp.valueOf(s(2).stripPrefix(" ").stripSuffix(" ")), charttime = Timestamp.valueOf(s(3).stripPrefix(" ").stripSuffix(" ")),
        category = s(5), intime = Timestamp.valueOf(s(6).stripPrefix(" ").stripSuffix(" ")), text = s(7))))

    val minICU: DataFrame = icu.groupBy("subject_id").agg(min("intime")).toDF("subject_id", "mintime")

    val joinICUBack: DataFrame = minICU.join(icu, "subject_id").drop("subject_id")

    val filteredBack = joinICUBack.filter(joinICUBack("mintime") === joinICUBack("intime"))

    val noteJoin = notes.join(filteredBack, "hadm_id").drop("row_id")

    val filteredNotes = noteJoin.filter((noteJoin("intime") + expr("INTERVAL 36 HOUR") >
      noteJoin("charttime")) && (noteJoin("intime") + expr("INTERVAL  HOUR") < noteJoin("outtime")))
    val features = Topics.run(sqlContext, filteredNotes, 30, 500)

    val adm = sqlContext.load("jdbc", Map("driver" -> "org.postgresql.Driver","url" -> "jdbc:postgresql://mimic3-1.coh8b3jmul4y.us-west-2.rds.amazonaws.com:5432/MIMIC?user=jfund&password=G0tbeared",
      "dbtable" -> "mimiciii.admissions"))

    // get target function
    val toStringToOne = udf((t: String) => if (t == null) 0.0 else 1.0)
    val genderToValue = udf((t: String) => if (t.equals("M")) 0.0 else 1.0)

    val admDF = adm.withColumn("deathtime", toStringToOne(adm("deathtime")))
    val notDead = admDF.filter(admDF("deathtime") === 0)
    admDF.show()
    val joinedAdmFeatures = features.join(admDF, "hadm_id")
    val vectored = joinedAdmFeatures.select("deathtime","features").map(row => LabeledPoint(row.getDouble(0), Vectors.dense(row.getSeq(1).toArray[Double])))
    // Split data into training (60%) and test (40%).
    val splits = vectored.randomSplit(Array(0.7, 0.3), seed = 11L)
    val training = splits(0).cache()
    val testing = splits(1)
    // Run training algorithm to build the model
    val numIterations = 100
    val model = SVMWithSGD.train(training, numIterations)
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
    println("Area under ROC = " + auROC)

  }

  def createContext(appName: String, masterUrl: String): SparkContext = {
    val conf = new SparkConf().setAppName(appName).setMaster(masterUrl).setExecutorEnv("memory", "15g")
    new SparkContext(conf)


  }
}

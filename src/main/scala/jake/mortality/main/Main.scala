package jake.mortality.main


import java.sql.Timestamp


import jake.mortality.base.Base
import jake.mortality.dynamic.Dynamic
import jake.mortality.topics.Topics
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

    Dynamic.run(sqlContext, "24", 35)
    Base.run(sqlContext)

  }

  def createContext(appName: String, masterUrl: String): SparkContext = {
    val conf = new SparkConf().setAppName(appName).setMaster(masterUrl).setExecutorEnv("memory", "15g")
    new SparkContext(conf)


  }
}

package jake.mortality.main


import java.sql.Timestamp


import model.Note
import jake.mortality.tdf.TdfText
import jake.mortality.lin.{toBreezeVector, toBreezeMatrix}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, Tokenizer}
import org.apache.spark.mllib.clustering.{LDA, DistributedLDAModel}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{DenseVector,Vectors, SparseVector}
import org.apache.spark.rdd.RDD

import org.apache.spark.sql.types.{IntegerType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.postgresql
import org.apache.spark.sql.functions._





/**
  * Created by jake on 4/14/16.
  */
object Main  {


  def main(args: Array[String]) {



    val sc = createContext("final", "local")
    val sqlContext = new SQLContext(sc)
    val testDF = sc.textFile("/Users/jake/dataFinal/output.txt").map(row => row.split('|').map(s => s.replace("  ", ""))).collect
    testDF(0) = testDF(0).map(s => s.replace(" ", ""))
    var finishedArray = List[Array[String]]()
    var cur = 0
    var arrcur = 0

    var i = 0
    for  ( i <- testDF.indices) {

      val current = testDF(i)

      if (current.last.size > 0 && current.last.charAt(current.last.size - 1).equals('+')) {

        testDF(cur).update(testDF(cur).length - 1, testDF(cur).last.concat(current.last.replace("+"," ")))
      }
      else {

        finishedArray = finishedArray ::: List(testDF(cur))


        cur = i+1

      }

    }



    val dfs = finishedArray.toSeq.slice(2,finishedArray.size -3).map(s =>
      (s(0).replace(" ", "").toInt,s(1).replace(" ", "").toInt,Timestamp.valueOf(s(2).stripPrefix(" ").stripSuffix(" ")),Timestamp.valueOf(s(3).stripPrefix(" ").stripSuffix(" ")),
         s(5), Timestamp.valueOf(s(6).stripPrefix(" ").stripSuffix(" ")),s(7)))


    val icu = sqlContext.load("jdbc", Map("driver" -> "org.postgresql.Driver","url" -> "jdbc:postgresql://mimic3-1.coh8b3jmul4y.us-west-2.rds.amazonaws.com:5432/MIMIC?user=jfund&password=G0tbeared","dbtable" -> "MIMICIII.ICUSTAYS"))

    val notes = sqlContext.load("jdbc", Map("driver" -> "org.postgresql.Driver","url" -> "jdbc:postgresql://mimic3-1.coh8b3jmul4y.us-west-2.rds.amazonaws.com:5432/MIMIC?user=jfund&password=G0tbeared","dbtable" -> "MIMICIII.NOTEEVENTS"))

   val test = sqlContext.createDataFrame(finishedArray.slice(2,finishedArray.size -3).map(f = s =>
     Note(subject_id = s(0).replace(" ", "").toInt, hadm_id = s(1).replace(" ", "").toInt, chartdate = Timestamp.valueOf(s(2).stripPrefix(" ").stripSuffix(" ")), charttime = Timestamp.valueOf(s(3).stripPrefix(" ").stripSuffix(" ")),
       category = s(5), intime = Timestamp.valueOf(s(6).stripPrefix(" ").stripSuffix(" ")), text = s(7))))





     val minICU: DataFrame = test.groupBy("subject_id").agg(min("intime")).toDF("subject_id","mintime")




    val joinICUBack: DataFrame = minICU.join(test, "subject_id").drop("subject_id")

    val filteredBack = joinICUBack.filter(joinICUBack("mintime") === joinICUBack("intime"))






    val grouped = filteredBack.select("hadm_id","text").rdd.groupBy(s => s.getInt(0)).collect()
    val stopWords = sc.textFile("/Users/jake/dataFinal/stopwords").collect()

    val text = grouped.map(s => (s._1,TdfText.run(sqlContext,s, 500, stopWords)))
    val vocabulary = text.flatMap(s=> s._2.map(l => l._2)).distinct

    val vocabInd: Map[String, Int]  = vocabulary.zipWithIndex.toMap
    val regexpr = """[a-zA-Z]+""".r
    val shaveText = filteredBack.select("text").map(row => regexpr.findAllIn(row.getString(0)).toSeq)

    val unionTextZip = vocabulary.zipWithIndex.toMap

    val documents =
      shaveText.zipWithIndex.map { case (tokens, id) =>
        val counts = new scala.collection.mutable.HashMap[Int, Double]()
        tokens.foreach { term =>
          if (vocabulary.contains(term)) {
            val idx = vocabInd(term)
            counts(idx) = counts.getOrElse(idx, 0.0) + 1.0
          }
        }
        (id, Vectors.sparse(vocabulary.size, counts.toSeq))
      }


    // Cluster the documents into three topics using LDA
    val ldaModel = new LDA().setK(10).setMaxIterations(1).run(documents)
    // Output topics. Each is a distribution over words (matching word count vectors)
    // Print topics, showing top-weighted 10 terms for each topic.

    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 10)
    topicIndices.foreach { case (terms, termWeights) =>
      println("TOPIC:")
      terms.zip(termWeights).foreach { case (term, weight) =>
        println(s"${vocabulary(term.toInt)}\t$weight")
      }
      println()
    }
    val transp = ldaModel.topicsMatrix.transpose
    println(transp)

    val returns = documents.map(s => (s._1, transp.multiply(s._2)))

    val docToPatient = filteredBack.select("hadm_id").map(s => s.getInt(0)).zipWithIndex.map(_.swap).collect().toMap

    println(docToPatient.toArray.to)
    val groupNotes = returns.map(s => (docToPatient(s._1.toInt),s._2)).groupBy(s=> s._1)
    val preFeats = groupNotes.map(s => (s._1, s._2.reduce((s1,s2) =>
      (s1._1, new DenseVector(new DenseVector(s1._2.toArray.zip(s2._2.toArray).map(f1 => f1._1.toDouble + f1._2.toDouble)).values.map(d => d/s._2.size.toDouble))))))
    println(preFeats.collect().to)
}



  def createContext(appName: String, masterUrl: String): SparkContext = {
    val conf = new SparkConf().setAppName(appName).setMaster(masterUrl).setExecutorEnv("memory","15g")
    new SparkContext(conf)

    /Users/jake/mortality
  }
}

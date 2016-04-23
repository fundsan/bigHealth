package jake.mortality.jake.mortality.topics

import jake.mortality.tdf.TdfText
import model.hadmAndFeatures
import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.linalg.{DenseVector, Vectors}
import org.apache.spark.sql.{Row, DataFrame, SQLContext}

/**
  * Created by jake on 4/14/16.
  */
object Topics {
  def run(sqlContext: SQLContext, df: DataFrame, numTopics: Int, numWords: Int): DataFrame = {
    val sc = sqlContext.sparkContext
    val grouped = df.select("hadm_id", "text").rdd.groupBy(s => s.getInt(0)).collect()
    sc.hadoopConfiguration.set("fs.s3n.awsAccessKeyId", "*")
    sc.hadoopConfiguration.set("fs.s3n.awsSecretAccessKey", "*")
    val stopWords = sc.textFile("s3://jakemimc/stopwords").collect()
    val text = grouped.map(s => (s._1, TdfText.run(sqlContext, s, numWords, stopWords)))
    val vocabulary = text.flatMap(s => s._2.map(l => l._2)).distinct
    val vocabInd: Map[String, Int] = vocabulary.zipWithIndex.toMap
    val regexpr = """[a-zA-Z]+""".r
    val shaveText = df.select("text").map(row => regexpr.findAllIn(row.getString(0)).toSeq)
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
    val ldaModel = new LDA().setK(numTopics).setMaxIterations(1).run(documents)
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
    val returns = documents.map(s => (s._1, transp.multiply(s._2)))
    val docToPatient = df.select("hadm_id").map(s => s.getInt(0)).zipWithIndex.map(_.swap).collect().toMap
    val groupNotes = returns.map(s => (docToPatient(s._1.toInt), s._2)).groupBy(s => s._1)
    val preFeats = groupNotes.map(s => (s._1, s._2.reduce((s1, s2) =>
      (s1._1, new DenseVector(new DenseVector(s1._2.toArray.zip(s2._2.toArray).map(f1 => f1._1.toDouble + f1._2.toDouble)).values.map(d => d / s._2.size.toDouble))))))
    val features = preFeats.map(s => s._2).map(s => (s._1,s._2.toArray.toSeq)).map(s => hadmAndFeatures(s._1,s._2))
    val newDF = sqlContext.createDataFrame(
      features)
    newDF
  }

}

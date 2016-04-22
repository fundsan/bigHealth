package jake.mortality.tdf
import org.apache.spark.ml.feature.{Tokenizer, CountVectorizer, CountVectorizerModel}
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.ScalaReflection.Schema
import org.apache.spark.sql.catalyst.expressions.Log
import org.apache.spark.sql.types.{IntegerType, DataType, StructField, StructType}
import org.apache.spark.sql.{Row, DataFrame, SQLContext}
import org.apache.spark.mllib.linalg.Vectors

import scala.collection.immutable.HashMap

/**
  * Created by jake on 4/14/16.
  */
object TdfText {
  def run(sqlContext: SQLContext, itDf: (Int, Iterable[org.apache.spark.sql.Row]), n: Int, stopWords: Array[String]): List[(Double,String)] = {

    val hadm_id = itDf._1

    val N = itDf._2.size
    val regexpr = """[a-zA-Z]+""".r
    val wordsToDoc = scala.collection.mutable.Map[String, Integer]()
    val words = itDf._2.map(row => regexpr.findAllIn(row.getString(1)).toList.map(s => s.toLowerCase()))
    val textWordMap = words.map { words =>
      val wordMap = scala.collection.mutable.Map[String, Integer]()
      words.foreach(s =>
        if (!wordMap.contains(s)) {
          wordMap(s) = 1
          if (!wordsToDoc.contains(s))
            wordsToDoc(s) = 1
          wordsToDoc(s) =  wordsToDoc(s) + 1
        }
        else
          wordMap(s) = wordMap(s) + 1


      )
      wordMap
    }

    val AllWord = words.flatten.toSet.--(stopWords.toSet)
    val valuesForWords = AllWord.map { s =>
      val idf = 1.0+ math.log(N / wordsToDoc(s).toDouble)
      val sum = textWordMap.map { words =>

        var amount = 0.0
        if (words.contains(s)) {

          amount = words(s) * idf

        }

        amount
      }.sum
      (sum, s)
    }.toList.sortBy(st => -st._1)

    valuesForWords.take(n)

  }
}
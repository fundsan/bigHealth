package model

import java.sql.Timestamp

/**
  * Created by jake on 4/19/16.
  */
case class Note(subject_id: Int, hadm_id: Int, chartdate: Timestamp, charttime: Timestamp, category: String, intime: Timestamp, text: String)
case class hadmAndFeatures(hadm_id: Int, features: Seq[Double])
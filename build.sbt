name := "mortality"

version := "1.0"

scalaVersion :="2.10.4"

resolvers ++= Seq(
  Resolver.sonatypeRepo("releases"),
  Resolver.sonatypeRepo("snapshots")
)

libraryDependencies ++= Seq(
  "org.apache.spark"  % "spark-core_2.10"              % "1.6.1" % "provided",
  "org.apache.spark"  % "spark-mllib_2.10"             % "1.6.1",
  "org.apache.spark"  % "spark-graphx_2.10"            % "1.6.1",
  "com.databricks"    % "spark-csv_2.10"               % "1.3.0",
  "org.postgresql"    % "postgresql"                   % "9.4.1207.jre7",
  "joda-time" 		  % "joda-time" 				   % "2.9.3"
)

libraryDependencies += "org.scalatest" % "scalatest_2.10" % "2.2.1" % "test"

javaOptions += "-Xmx8G"
//…

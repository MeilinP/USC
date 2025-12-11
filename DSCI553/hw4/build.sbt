name := "hw4"

version := "1.0"

scalaVersion := "2.12.10"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.1.2" % "provided",
  "org.apache.spark" %% "spark-sql" % "3.1.2" % "provided",
  "org.apache.spark" %% "spark-graphx" % "3.1.2" % "provided",
  "graphframes" % "graphframes" % "0.8.2-spark3.1-s_2.12"
)

resolvers += "Spark Packages Repo" at "https://repos.spark-packages.org/"

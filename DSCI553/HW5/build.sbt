name := "hw5"

version := "1.0"

scalaVersion := "2.12.15"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.1.2" % "provided",
  "org.apache.spark" %% "spark-sql" % "3.1.2" % "provided"
)

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}

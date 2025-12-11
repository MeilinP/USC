name := "hw3"
version := "1.0"
scalaVersion := "2.12.15"

lazy val sparkVer = "3.1.2"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core"  % sparkVer % "provided",
  "org.apache.spark" %% "spark-sql"   % sparkVer % "provided",
  "org.apache.spark" %% "spark-mllib" % sparkVer % "provided"
)

enablePlugins(sbtassembly.AssemblyPlugin)

import sbtassembly.AssemblyPlugin.autoImport._
assembly / test := {}
assembly / assemblyMergeStrategy := {
  case "META-INF/services/org.apache.spark.sql.sources.DataSourceRegister" => MergeStrategy.filterDistinctLines
  case PathList("META-INF", _ @ _*) => MergeStrategy.discard
  case "module-info.class" => MergeStrategy.discard
  case _ => MergeStrategy.first
}
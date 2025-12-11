name := "hw2"

version := "1.0"

scalaVersion := "2.12.10"

libraryDependencies += "org.apache.spark" %% "spark-core" % "3.1.2" % "provided"
Compile / packageBin / artifactPath := baseDirectory.value / ".." / "hw2.jar"

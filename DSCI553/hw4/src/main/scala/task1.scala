import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.sql.functions._
import org.graphframes.GraphFrame

object task1 {
  def main(args: Array[String]): Unit = {
    val filterThreshold = args(0).toInt
    val inputFilePath = args(1)
    val communityOutputPath = args(2)
    
    val spark = SparkSession
      .builder()
      .appName("task1")
      .master("local[*]")
      .getOrCreate()
    
    val sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    
    import spark.implicits._
    
    val data = sc.textFile(inputFilePath)
    val header = data.first()
    val records = data.filter(line => line != header)
      .map(line => {
        val parts = line.split(",")
        (parts(0), parts(1))
      })
    
    val userBusinesses = records.groupByKey()
      .mapValues(_.toSet)
      .collectAsMap()
    
    val allUsers = userBusinesses.keys.toList.sorted
    
    val edges = scala.collection.mutable.ArrayBuffer[(String, String)]()
    
    for (i <- allUsers.indices) {
      for (j <- i + 1 until allUsers.length) {
        val user1 = allUsers(i)
        val user2 = allUsers(j)
        
        if (userBusinesses.contains(user1) && userBusinesses.contains(user2)) {
          val commonBusinesses = userBusinesses(user1).intersect(userBusinesses(user2))
          
          if (commonBusinesses.size >= filterThreshold) {
            edges += ((user1, user2))
            edges += ((user2, user1))
          }
        }
      }
    }
    
    val nodesWithEdges = edges.flatMap(e => List(e._1, e._2)).toSet
    
    val vertices = spark.createDataFrame(
      nodesWithEdges.toSeq.map(Tuple1(_))
    ).toDF("id")
    
    val edgesDF = spark.createDataFrame(edges).toDF("src", "dst")
    
    val graph = GraphFrame(vertices, edgesDF)
    
    val result = graph.labelPropagation.maxIter(5).run()
    
    val communities = result.rdd
      .map(row => (row.getAs[Long]("label"), row.getAs[String]("id")))
      .groupByKey()
      .map { case (_, users) => users.toList.sorted }
      .collect()
    
    val sortedCommunities = communities.sortBy(c => (c.size, c.head))
    
    val writer = new java.io.PrintWriter(communityOutputPath)
    try {
      for (community <- sortedCommunities) {
        writer.println(community.map(u => s"'$u'").mkString(", "))
      }
    } finally {
      writer.close()
    }
    
    spark.stop()
  }
}
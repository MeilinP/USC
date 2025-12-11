import org.apache.spark.{SparkConf, SparkContext}
import scala.collection.mutable

object task1 {
  
  implicit val listOrdering: Ordering[List[String]] = Ordering.by((_: List[String]).mkString(","))
  
  def apriori(baskets: Iterator[Set[String]], support: Double): Iterator[List[String]] = {
    val basketList = baskets.toList
    if (basketList.isEmpty) {
      return Iterator.empty
    }
    
    val itemCounts = mutable.Map[String, Int]()
    for (basket <- basketList; item <- basket) {
      itemCounts(item) = itemCounts.getOrElse(item, 0) + 1
    }
    
    val frequentSingles = itemCounts.filter(_._2 >= support).keys.toSet
    var resultItemsets = mutable.ListBuffer[List[String]]()
    
    for (item <- frequentSingles.toList.sorted) {
      resultItemsets += List(item)
    }
    
    if (frequentSingles.size < 2) {
      return resultItemsets.iterator
    }
    
    val sortedSingles = frequentSingles.toList.sorted
    var candidatePairs = mutable.Map[List[String], Int]()
    
    for (i <- sortedSingles.indices; j <- i + 1 until sortedSingles.length) {
      candidatePairs(List(sortedSingles(i), sortedSingles(j))) = 0
    }
    
    for (basket <- basketList; pair <- candidatePairs.keys) {
      if (pair.forall(basket.contains)) {
        candidatePairs(pair) += 1
      }
    }
    
    val frequentPairs = candidatePairs.filter(_._2 >= support).keys.toSet
    resultItemsets ++= frequentPairs.toList.sorted(listOrdering)
    
    if (frequentPairs.isEmpty) {
      return resultItemsets.iterator
    }
    
    val filteredBaskets = basketList.map(basket => 
      basket.filter(frequentSingles.contains).toList
    ).filter(_.length >= 3)
    
    var currentFrequent = frequentPairs
    var k = 3
    
    while (currentFrequent.nonEmpty) {
      val allItems = currentFrequent.flatten.toSet
      val candidatesK = allItems.toList.sorted.combinations(k).toList
      
      if (candidatesK.isEmpty) {
        return resultItemsets.iterator
      }
      
      var candidateCounts = mutable.Map[List[String], Int]()
      for (candidate <- candidatesK) {
        candidateCounts(candidate) = 0
      }
      
      for (basket <- filteredBaskets; candidate <- candidatesK) {
        if (candidate.forall(basket.contains)) {
          candidateCounts(candidate) += 1
        }
      }
      
      val nextFrequent = candidateCounts.filter(_._2 >= support).keys.toSet
      if (nextFrequent.isEmpty) {
        return resultItemsets.iterator
      }
      
      resultItemsets ++= nextFrequent.toList.sorted
      currentFrequent = nextFrequent
      k += 1
    }
    
    resultItemsets.iterator
  }
  
  def countCandidates(baskets: Iterator[Set[String]], candidates: Set[List[String]]): Iterator[(List[String], Int)] = {
    val basketList = baskets.toList
    val counts = mutable.Map[List[String], Int]()
    
    for (basket <- basketList; candidate <- candidates) {
      if (candidate.forall(basket.contains)) {
        counts(candidate) = counts.getOrElse(candidate, 0) + 1
      }
    }
    
    counts.iterator
  }
  
  def formatOutput(itemsets: List[List[String]]): String = {
    val grouped = itemsets.groupBy(_.length).toList.sortBy(_._1)
    val lines = mutable.ListBuffer[String]()
    
    for ((size, items) <- grouped) {
      val sortedItems = items.sorted
      val formatted = if (size == 1) {
        sortedItems.map(item => s"('${item.head}')").mkString(",")
      } else {
        sortedItems.map(item => s"(${item.map(x => s"'$x'").mkString(",")})").mkString(",")
      }
      lines += formatted
    }
    
    lines.mkString("\n\n")
  }
  
  def main(args: Array[String]): Unit = {
    val startTime = System.currentTimeMillis()
    
    val caseNum = args(0).toInt
    val support = args(1).toInt
    val inputFile = args(2)
    val outputFile = args(3)
    
    val conf = new SparkConf().setAppName("task1")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    
    val data = sc.textFile(inputFile)
    val header = data.first()
    val pairs = data.filter(_ != header).map(_.split(","))
    
    val baskets = if (caseNum == 1) {
      pairs.map(arr => (arr(0), arr(1)))
           .groupByKey()
           .map { case (_, businesses) => businesses.toSet }
    } else {
      pairs.map(arr => (arr(1), arr(0)))
           .groupByKey()
           .map { case (_, users) => users.toSet }
    }
    
    baskets.cache()
    
    val numPartitions = baskets.getNumPartitions
    val partitionSupport = support.toDouble / numPartitions
    
    val candidates = baskets.mapPartitions(part => apriori(part, partitionSupport))
                           .distinct()
                           .collect()
                           .toList
    
    val candidatesOutput = formatOutput(candidates)
    
    val candidateSet = candidates.toSet
    val candidateBroadcast = sc.broadcast(candidateSet)
    
    val frequent = baskets.mapPartitions(part => countCandidates(part, candidateBroadcast.value))
                         .reduceByKey(_ + _)
                         .filter(_._2 >= support)
                         .keys
                         .collect()
                         .toList
    
    val frequentOutput = formatOutput(frequent)
    
    val writer = new java.io.PrintWriter(outputFile)
    try {
      writer.write("Candidates:\n")
      writer.write(candidatesOutput)
      writer.write("\n\n")
      writer.write("Frequent Itemsets:\n")
      writer.write(frequentOutput)
    } finally {
      writer.close()
    }
    
    baskets.unpersist()
    sc.stop()
    
    val duration = (System.currentTimeMillis() - startTime) / 1000
    println(s"Duration: $duration")
  }
}
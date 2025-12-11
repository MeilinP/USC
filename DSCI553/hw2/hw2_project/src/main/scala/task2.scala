import org.apache.spark.{SparkConf, SparkContext}
import scala.collection.mutable

object task2 {
  
  implicit val listOrdering: Ordering[List[String]] = Ordering.by((_: List[String]).mkString(","))
  
  def pcyFrequentItemsets(baskets: Iterator[Set[String]], totalBaskets: Long, 
                          minSupport: Int, hashBuckets: Int): Iterator[List[String]] = {
    val basketList = baskets.toList
    if (basketList.isEmpty) {
      return Iterator.empty
    }
    
    val localRatio = basketList.length.toDouble / totalBaskets
    val localThreshold = localRatio * minSupport
    
    val itemFrequency = mutable.Map[String, Int]()
    val bucketCounts = Array.fill(hashBuckets)(0)
    
    for (basket <- basketList) {
      for (item <- basket) {
        itemFrequency(item) = itemFrequency.getOrElse(item, 0) + 1
      }
      
      val basketItems = basket.toList
      for (i <- basketItems.indices; j <- i + 1 until basketItems.length) {
        val bucketId = Math.abs((basketItems(i).hashCode ^ basketItems(j).hashCode) % hashBuckets)
        bucketCounts(bucketId) += 1
      }
    }
    
    val frequentBuckets = bucketCounts.zipWithIndex
                                     .filter(_._1 >= localThreshold)
                                     .map(_._2)
                                     .toSet
    
    val frequentSingles = itemFrequency.filter(_._2 >= localThreshold).keys.toSet
    val resultItemsets = mutable.ListBuffer[List[String]]()
    
    for (item <- frequentSingles.toList.sorted) {
      resultItemsets += List(item)
    }
    
    if (frequentSingles.size < 2) {
      return resultItemsets.iterator
    }
    
    val sortedSingles = frequentSingles.toList.sorted
    val candidatePairs = mutable.Map[List[String], Int]()
    
    for (i <- sortedSingles.indices; j <- i + 1 until sortedSingles.length) {
      val item1 = sortedSingles(i)
      val item2 = sortedSingles(j)
      val bucketId = Math.abs((item1.hashCode ^ item2.hashCode) % hashBuckets)
      
      if (frequentBuckets.contains(bucketId)) {
        candidatePairs(List(item1, item2)) = 0
      }
    }
    
    for (basket <- basketList; pair <- candidatePairs.keys) {
      if (pair.forall(basket.contains)) {
        candidatePairs(pair) += 1
      }
    }
    
    val frequentPairs = candidatePairs.filter(_._2 >= localThreshold).keys.toSet
    resultItemsets ++= frequentPairs.toList.sorted
    
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
      
      val candidateCounts = mutable.Map[List[String], Int]()
      for (candidate <- candidatesK) {
        candidateCounts(candidate) = 0
      }
      
      for (basket <- filteredBaskets; candidate <- candidatesK) {
        if (candidate.forall(basket.contains)) {
          candidateCounts(candidate) += 1
        }
      }
      
      val nextFrequent = candidateCounts.filter(_._2 >= localThreshold).keys.toSet
      if (nextFrequent.isEmpty) {
        return resultItemsets.iterator
      }
      
      resultItemsets ++= nextFrequent.toList.sorted
      currentFrequent = nextFrequent
      k += 1
    }
    
    resultItemsets.iterator
  }
  
  def countGlobalSupport(baskets: Iterator[Set[String]], 
                        candidates: Set[List[String]]): Iterator[(List[String], Int)] = {
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
      val sortedItems = items.map(_.sorted).distinct.sorted
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
    
    val filterThreshold = args(0).toInt
    val supportThreshold = args(1).toInt
    val inputPath = args(2)
    val outputPath = args(3)
    
    val conf = new SparkConf().setAppName("task2_pcy")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    
    val rawData = sc.textFile(inputPath)
    val header = rawData.first()
    
    def extractTransaction(line: String): Option[(String, String)] = {
      try {
        val fields = line.replace("\"", "").split(",")
        if (fields.length < 6) return None
        
        val date = fields(0)
        val customer = fields(1)
        val product = fields(5)
        
        val basketId = s"$date-$customer"
        val productId = product.toDouble.toInt.toString
        
        Some((basketId, productId))
      } catch {
        case _: Exception => None
      }
    }
    
    val transactions = rawData.filter(_ != header)
                             .map(extractTransaction)
                             .filter(_.isDefined)
                             .map(_.get)
    
    val baskets = transactions.groupByKey()
                             .mapValues(products => products.toSet)
                             .filter { case (_, products) => products.size > filterThreshold }
                             .values
    
    baskets.cache()
    
    val totalBaskets = baskets.count()
    val hashBuckets = 2000
    
    val candidates = baskets.mapPartitions(part => 
      pcyFrequentItemsets(part, totalBaskets, supportThreshold, hashBuckets)
    ).distinct().collect().toList
    
    val candidatesOutput = formatOutput(candidates)
    
    val candidateSet = candidates.toSet
    val candidateBroadcast = sc.broadcast(candidateSet)
    
    val frequent = baskets.mapPartitions(part => 
      countGlobalSupport(part, candidateBroadcast.value)
    ).reduceByKey(_ + _)
     .filter(_._2 >= supportThreshold)
     .keys
     .collect()
     .toList
    
    val frequentOutput = formatOutput(frequent)
    
    val writer = new java.io.PrintWriter(outputPath)
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
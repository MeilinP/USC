import org.apache.spark.{SparkConf, SparkContext}
import scala.collection.mutable
import scala.util.Random

object task {
  
  case class ClusterStats(N: Int, SUM: Array[Double], SUMSQ: Array[Double])
  
  def loadData(sc: SparkContext, filename: String): Array[(Int, Array[Double])] = {
    sc.textFile(filename)
      .map { line =>
        val parts = line.split(",")
        val idx = parts(0).toInt
        val features = parts.drop(2).map(_.toDouble)
        (idx, features)
      }
      .collect()
  }
  
  def mahalanobisDistance(point: Array[Double], stats: ClusterStats, d: Int): Double = {
    if (stats.N == 0) return Double.PositiveInfinity
    
    val centroid = stats.SUM.map(_ / stats.N)
    val variance = stats.SUMSQ.zip(stats.SUM).map { case (sumsq, sum) =>
      Math.max(sumsq / stats.N - Math.pow(sum / stats.N, 2), 1e-10)
    }
    
    val diff = point.zip(centroid).map { case (p, c) => p - c }
    val distance = Math.sqrt(diff.zip(variance).map { case (d, v) => d * d / v }.sum)
    distance
  }
  
  def mergeStats(s1: ClusterStats, s2: ClusterStats): ClusterStats = {
    ClusterStats(
      s1.N + s2.N,
      s1.SUM.zip(s2.SUM).map { case (a, b) => a + b },
      s1.SUMSQ.zip(s2.SUMSQ).map { case (a, b) => a + b }
    )
  }
  
  def updateStats(stats: ClusterStats, point: Array[Double]): ClusterStats = {
    ClusterStats(
      stats.N + 1,
      stats.SUM.zip(point).map { case (s, p) => s + p },
      stats.SUMSQ.zip(point).map { case (s, p) => s + p * p }
    )
  }
  
  def kMeans(data: Array[Array[Double]], k: Int, maxIter: Int = 100): Array[Int] = {
    val n = data.length
    val d = data(0).length
    val random = new Random(42)
    
    val initialIndices = random.shuffle((0 until n).toList).take(k)
    var centroids = initialIndices.map(i => data(i)).toArray
    
    var labels = Array.fill(n)(0)
    var changed = true
    var iter = 0
    
    while (changed && iter < maxIter) {
      changed = false
      
      for (i <- 0 until n) {
        val point = data(i)
        var minDist = Double.PositiveInfinity
        var bestCluster = 0
        
        for (j <- 0 until k) {
          val dist = Math.sqrt(point.zip(centroids(j)).map { case (a, b) => Math.pow(a - b, 2) }.sum)
          if (dist < minDist) {
            minDist = dist
            bestCluster = j
          }
        }
        
        if (labels(i) != bestCluster) {
          labels(i) = bestCluster
          changed = true
        }
      }
      
      for (j <- 0 until k) {
        val clusterPoints = (0 until n).filter(i => labels(i) == j).map(i => data(i))
        if (clusterPoints.nonEmpty) {
          centroids(j) = clusterPoints.reduce((a, b) => a.zip(b).map { case (x, y) => x + y })
            .map(_ / clusterPoints.length)
        }
      }
      
      iter += 1
    }
    
    labels
  }
  
  def main(args: Array[String]): Unit = {
    val inputFile = args(0)
    val nCluster = args(1).toInt
    val outputFile = args(2)
    
    val conf = new SparkConf().setAppName("BFR")
    val sc = new SparkContext(conf)
    
    val allData = loadData(sc, inputFile)
    val nPoints = allData.length
    val d = allData(0)._2.length
    
    val random = new Random(42)
    val shuffledIndices = random.shuffle((0 until nPoints).toList).toArray
    
    val DS = mutable.Map[Int, ClusterStats]()
    val CS = mutable.Map[Int, ClusterStats]()
    val RS = mutable.Map[Int, Array[Double]]()
    val CSPoints = mutable.Map[Int, Int]()
    val pointToDS = mutable.Map[Int, Int]()
    
    val intermediateResults = mutable.ArrayBuffer[(Int, Int, Int, Int)]()
    val chunkSize = nPoints / 5
    
    val firstChunkIndices = shuffledIndices.take(chunkSize)
    val firstChunk = firstChunkIndices.map(i => allData(i))
    val firstChunkFeatures = firstChunk.map(_._2)
    
    val largeK = 5 * nCluster
    val labelsLarge = kMeans(firstChunkFeatures, largeK)
    
    val labelCounts = labelsLarge.groupBy(identity).mapValues(_.length)
    
    val remainingPoints = mutable.ArrayBuffer[(Int, Array[Double])]()
    val remainingFeatures = mutable.ArrayBuffer[Array[Double]]()
    
    for (i <- firstChunk.indices) {
      val (idx, features) = firstChunk(i)
      if (labelCounts(labelsLarge(i)) == 1) {
        RS(idx) = features
      } else {
        remainingPoints += ((idx, features))
        remainingFeatures += features
      }
    }
    
    if (remainingFeatures.nonEmpty) {
      val labelsN = kMeans(remainingFeatures.toArray, nCluster)
      
      for (clusterId <- 0 until nCluster) {
        DS(clusterId) = ClusterStats(0, Array.fill(d)(0.0), Array.fill(d)(0.0))
      }
      
      for (i <- remainingPoints.indices) {
        val (idx, features) = remainingPoints(i)
        val clusterId = labelsN(i)
        pointToDS(idx) = clusterId
        DS(clusterId) = updateStats(DS(clusterId), features)
      }
    }
    
    if (RS.size > 1) {
      val rsIndices = RS.keys.toArray
      val rsFeatures = rsIndices.map(RS(_))
      
      val kRs = Math.min(5 * nCluster, rsFeatures.length)
      val labelsRs = kMeans(rsFeatures, kRs)
      
      val rsLabelCounts = labelsRs.groupBy(identity).mapValues(_.length)
      
      val newRS = mutable.Map[Int, Array[Double]]()
      val labelToCS = mutable.Map[Int, Int]()
      var nextCSId = 0
      
      for (i <- rsIndices.indices) {
        val idx = rsIndices(i)
        if (rsLabelCounts(labelsRs(i)) == 1) {
          newRS(idx) = RS(idx)
        } else {
          val label = labelsRs(i)
          if (!labelToCS.contains(label)) {
            labelToCS(label) = nextCSId
            CS(nextCSId) = ClusterStats(0, Array.fill(d)(0.0), Array.fill(d)(0.0))
            nextCSId += 1
          }
          
          val csId = labelToCS(label)
          CSPoints(idx) = csId
          CS(csId) = updateStats(CS(csId), RS(idx))
        }
      }
      
      RS.clear()
      RS ++= newRS
    }
    
    val numDiscard = DS.values.map(_.N).sum
    val numCSClusters = CS.size
    val numCompression = CS.values.map(_.N).sum
    val numRS = RS.size
    intermediateResults += ((numDiscard, numCSClusters, numCompression, numRS))
    
    for (roundNum <- 1 until 5) {
      val startIdx = roundNum * chunkSize
      val endIdx = Math.min((roundNum + 1) * chunkSize, nPoints)
      val chunkIndices = shuffledIndices.slice(startIdx, endIdx)
      
      val newPoints = chunkIndices.map(i => allData(i))
      
      val unassigned = mutable.ArrayBuffer[(Int, Array[Double])]()
      
      for ((idx, features) <- newPoints) {
        var bestDistance = Double.PositiveInfinity
        var bestCluster = -1
        
        for ((clusterId, stats) <- DS) {
          val dist = mahalanobisDistance(features, stats, d)
          if (dist < bestDistance) {
            bestDistance = dist
            bestCluster = clusterId
          }
        }
        
        if (bestDistance < 2 * Math.sqrt(d)) {
          pointToDS(idx) = bestCluster
          DS(bestCluster) = updateStats(DS(bestCluster), features)
        } else {
          unassigned += ((idx, features))
        }
      }
      
      val unassigned2 = mutable.ArrayBuffer[(Int, Array[Double])]()
      
      for ((idx, features) <- unassigned) {
        var bestDistance = Double.PositiveInfinity
        var bestCluster = -1
        
        for ((clusterId, stats) <- CS) {
          val dist = mahalanobisDistance(features, stats, d)
          if (dist < bestDistance) {
            bestDistance = dist
            bestCluster = clusterId
          }
        }
        
        if (bestDistance < 2 * Math.sqrt(d)) {
          CSPoints(idx) = bestCluster
          CS(bestCluster) = updateStats(CS(bestCluster), features)
        } else {
          unassigned2 += ((idx, features))
        }
      }
      
      for ((idx, features) <- unassigned2) {
        RS(idx) = features
      }
      
      if (RS.size > 1) {
        val rsIndices = RS.keys.toArray
        val rsFeatures = rsIndices.map(RS(_))
        
        val kRs = Math.min(5 * nCluster, rsFeatures.length)
        val labelsRs = kMeans(rsFeatures, kRs)
        
        val rsLabelCounts = labelsRs.groupBy(identity).mapValues(_.length)
        
        val newRS = mutable.Map[Int, Array[Double]]()
        val labelToCS = mutable.Map[Int, Int]()
        var nextCSId = if (CS.isEmpty) 0 else CS.keys.max + 1
        
        for (i <- rsIndices.indices) {
          val idx = rsIndices(i)
          if (rsLabelCounts(labelsRs(i)) == 1) {
            newRS(idx) = RS(idx)
          } else {
            val label = labelsRs(i)
            if (!labelToCS.contains(label)) {
              labelToCS(label) = nextCSId
              CS(nextCSId) = ClusterStats(0, Array.fill(d)(0.0), Array.fill(d)(0.0))
              nextCSId += 1
            }
            
            val csId = labelToCS(label)
            CSPoints(idx) = csId
            CS(csId) = updateStats(CS(csId), RS(idx))
          }
        }
        
        RS.clear()
        RS ++= newRS
      }
      
      val merged = mutable.Set[Int]()
      val csList = CS.keys.toArray
      
      for (i <- csList.indices) {
        for (j <- i + 1 until csList.length) {
          val c1 = csList(i)
          val c2 = csList(j)
          
          if (!merged.contains(c1) && !merged.contains(c2) && CS.contains(c1) && CS.contains(c2)) {
            val stats1 = CS(c1)
            val centroid1 = stats1.SUM.map(_ / stats1.N)
            val variance1 = stats1.SUMSQ.zip(stats1.SUM).map { case (sumsq, sum) =>
              Math.max(sumsq / stats1.N - Math.pow(sum / stats1.N, 2), 1e-10)
            }
            
            val stats2 = CS(c2)
            val centroid2 = stats2.SUM.map(_ / stats2.N)
            
            val diff = centroid2.zip(centroid1).map { case (a, b) => a - b }
            val dist = Math.sqrt(diff.zip(variance1).map { case (d, v) => d * d / v }.sum)
            
            if (dist < 2 * Math.sqrt(d)) {
              CS(c1) = mergeStats(CS(c1), CS(c2))
              for ((ptIdx, csId) <- CSPoints) {
                if (csId == c2) {
                  CSPoints(ptIdx) = c1
                }
              }
              CS.remove(c2)
              merged += c2
            }
          }
        }
      }
      
      if (roundNum == 4) {
        for (csId <- CS.keys.toList) {
          var bestDistance = Double.PositiveInfinity
          var bestDS = -1
          
          val csStats = CS(csId)
          val centroidCS = csStats.SUM.map(_ / csStats.N)
          
          for ((dsId, dsStats) <- DS) {
            val dist = mahalanobisDistance(centroidCS, dsStats, d)
            if (dist < bestDistance) {
              bestDistance = dist
              bestDS = dsId
            }
          }
          
          if (bestDistance < 2 * Math.sqrt(d)) {
            DS(bestDS) = mergeStats(DS(bestDS), CS(csId))
            for ((ptIdx, csIdPoint) <- CSPoints.toList) {
              if (csIdPoint == csId) {
                pointToDS(ptIdx) = bestDS
                CSPoints.remove(ptIdx)
              }
            }
            CS.remove(csId)
          }
        }
      }
      
      val numDiscard = DS.values.map(_.N).sum
      val numCSClusters = CS.size
      val numCompression = CS.values.map(_.N).sum
      val numRS = RS.size
      intermediateResults += ((numDiscard, numCSClusters, numCompression, numRS))
    }
    
    val finalAssignments = mutable.Map[Int, Int]()
    for ((idx, clusterId) <- pointToDS) {
      finalAssignments(idx) = clusterId
    }
    for (idx <- RS.keys) {
      finalAssignments(idx) = -1
    }
    for (idx <- CSPoints.keys) {
      finalAssignments(idx) = -1
    }
    
    val writer = new java.io.PrintWriter(new java.io.File(outputFile))
    writer.write("The intermediate results:\n")
    for (i <- intermediateResults.indices) {
      val (nd, ncc, nc, nr) = intermediateResults(i)
      writer.write(s"Round ${i + 1}: $nd,$ncc,$nc,$nr\n")
    }
    writer.write("\n")
    writer.write("The clustering results:\n")
    for (idx <- finalAssignments.keys.toSeq.sorted) {
      writer.write(s"$idx,${finalAssignments(idx)}\n")
    }
    writer.close()
    
    sc.stop()
  }
}

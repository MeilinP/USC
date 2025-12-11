import org.apache.spark.{SparkConf, SparkContext}
import scala.util.Random
import java.io.PrintWriter

object Task1 {
  def main(args: Array[String]): Unit = {
    val inputPath  = args(0)  
    val outputPath = args(1)

    val conf = new SparkConf().setAppName("task1_lsh_scala")
    val sc   = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val lines = sc.textFile(inputPath)
    val header = lines.first()
    val ub = lines.filter(_ != header).map(_.split(",")).map(x => (x(0), x(1)))

    val bizUsers = ub.map{ case(u,b) => (b,u) }
      .groupByKey()
      .mapValues(_.toSet)
      .cache()

    val userToIndex = ub.map(_._1).distinct().zipWithIndex().mapValues(_.toInt).collectAsMap()
    val userToIndexBc = sc.broadcast(userToIndex)
    val numUsers = userToIndex.size

    val numHashes = 60
    val rowsPerBand = 2
    val numBands = numHashes / rowsPerBand
    val largePrime = 1000000007L
    val rnd = new Random(42)

    def sampleCoeffs(n: Int, upTo: Int): Array[Int] = {
      val limit = math.max(upTo, n + 1)
      rnd.shuffle((1 until limit).toList).take(n).toArray
    }
    val a = sampleCoeffs(numHashes, math.max(2, numUsers))
    val b = sampleCoeffs(numHashes, math.max(2, numUsers))

    val signatures = bizUsers.map { case (biz, users) =>
      val idxs = users.iterator.map(u => userToIndexBc.value.getOrElse(u, -1)).filter(_ >= 0).toArray
      val sig = Array.fill[Int](numHashes)(Int.MaxValue)
      var h = 0
      while (h < numHashes) {
        var minv = Int.MaxValue
        val ah = a(h).toLong
        val bh = b(h).toLong
        var i = 0
        while (i < idxs.length) {
          val x = idxs(i).toLong
          val hv = (((ah * x + bh) % largePrime) % math.max(1, numUsers)).toInt
          if (hv < minv) minv = hv
          i += 1
        }
        sig(h) = if (minv == Int.MaxValue) 0 else minv
        h += 1
      }
      (biz, sig)
    }.cache()

    val bandPairs = signatures.flatMap { case (biz, sig) =>
      (0 until numBands).iterator.map { band =>
        val start = band * rowsPerBand
        val key = (band, (sig(start), sig(start + 1))) 
        (key, biz)
      }
    }
      .groupByKey()
      .values
      .flatMap { bizIter =>
        val arr = bizIter.toArray.distinct.sorted
        for {
          i <- arr.indices.iterator
          j <- (i + 1) until arr.length
        } yield (arr(i), arr(j))
      }
      .distinct()

    val bizUsersMap = bizUsers.collectAsMap()
    val bizUsersBc  = sc.broadcast(bizUsersMap)

    val candidates = bandPairs.map { case (b1, b2) =>
      val s1 = bizUsersBc.value.getOrElse(b1, Set.empty[String])
      val s2 = bizUsersBc.value.getOrElse(b2, Set.empty[String])
      val inter = (s1 intersect s2).size.toDouble
      val uni   = (s1 union s2).size.toDouble
      val jac   = if (uni == 0) 0.0 else inter / uni
      (b1, b2, jac)
    }
      .filter(_._3 >= 0.5)
      .collect()
      .sortBy{ case (b1, b2, _) => (b1, b2) }

    val out = new PrintWriter(outputPath)
    out.println("business_id_1,business_id_2,similarity") 
    candidates.foreach { case (b1, b2, s) =>
      out.println(s"$b1,$b2,${"%.6f".format(s)}")
    }
    out.close()

    sc.stop()
  }
}
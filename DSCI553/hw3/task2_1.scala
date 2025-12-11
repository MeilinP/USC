import org.apache.spark.{SparkConf, SparkContext}

object Task2_1 {
  private def pearson(a: Array[Double], b: Array[Double]): Double = {
    val n = a.length
    if (n == 0) return 0.0
    val ma = a.sum / n
    val mb = b.sum / n
    val num = (0 until n).map(i => (a(i) - ma) * (b(i) - mb)).sum
    val da  = math.sqrt((0 until n).map(i => math.pow(a(i) - ma, 2)).sum)
    val db  = math.sqrt((0 until n).map(i => math.pow(b(i) - mb, 2)).sum)
    if (da == 0.0 || db == 0.0) 0.0 else num / (da * db)
  }

  def main(args: Array[String]): Unit = {
    val trainPath = args(0)     
    val testPath  = args(1)     
    val outPath   = args(2)

    val conf = new SparkConf().setAppName("task2_1_item_based")
    val sc   = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val train = sc.textFile(trainPath)
    val head  = train.first()
    val rows  = train.filter(_ != head).map(_.split(",")).map(x => (x(0), x(1), x(2).toDouble))
    // (uid,bid,r)

    val busUsers = rows.map{case(u,b,r)=>(b,u)}.groupByKey().mapValues(_.toSet).collectAsMap()
    val userBus  = rows.map{case(u,b,r)=>(u,b)}.groupByKey().mapValues(_.toSet).collectAsMap()
    val busAvg   = rows.map{case(u,b,r)=>(b,r)}.groupByKey().mapValues(v=> v.sum/v.size).collectAsMap()
    val userAvg  = rows.map{case(u,b,r)=>(u,r)}.groupByKey().mapValues(v=> v.sum/v.size).collectAsMap()
    val busUserR = rows.map{case(u,b,r)=>(b,(u,r))}.groupByKey()
      .mapValues(it => it.toMap).collectAsMap()

    val test = sc.textFile(testPath)
    val thead = test.first()
    val pairs = test.filter(_ != thead).map(_.split(",")).map(x => (x(0), x(1))).collect() 

    val wCache = scala.collection.mutable.HashMap[(String,String), Double]()

    def w(b: String, b1: String): Double = {
      val key = if (b < b1) (b, b1) else (b1, b)
      wCache.getOrElseUpdate(key, {
        val uSet  = busUsers.getOrElse(b, Set.empty) intersect busUsers.getOrElse(b1, Set.empty)
        val n = uSet.size
        if (n <= 1) {
          (5.0 - math.abs(busAvg.getOrElse(b,3.5) - busAvg.getOrElse(b1,3.5))) / 5.0
        } else if (n == 2) {
          val two = uSet.toArray
          val a = (busUserR(b)(two(0)) - busUserR(b1)(two(0))).abs
          val c = (busUserR(b)(two(1)) - busUserR(b1)(two(1))).abs
          ((5.0 - a)/5.0 + (5.0 - c)/5.0) / 2.0
        } else {
          val common = uSet.toArray
          val r1 = common.map(u => busUserR(b)(u))
          val r2 = common.map(u => busUserR(b1)(u))
          pearson(r1, r2)
        }
      })
    }

    def predict(uid: String, bid: String): Double = {
      if (!userBus.contains(uid)) return 3.5
      if (!busUsers.contains(bid)) return userAvg.getOrElse(uid, 3.5)

      val candidates = userBus(uid).toArray.map { b1 =>
        val ww = w(bid, b1)
        val r  = busUserR(b1).getOrElse(uid, userAvg.getOrElse(uid,3.5))
        (ww, r)
      }.sortBy(-_._1).take(15)

      val num = candidates.map{case(wi,ri)=> wi*ri}.sum
      val den = candidates.map{case(wi,ri)=> math.abs(wi)}.sum
      if (den == 0.0) 3.5 else num/den
    }

    val preds = pairs.map{case(u,b)=> (u,b,predict(u,b))}.sortBy{case(u,b,_) => (u,b)}
    val out = new java.io.PrintWriter(outPath)
    out.println("user_id,business_id,prediction")
    preds.foreach{case(u,b,p)=> out.println(s"$u,$b,${math.max(1.0, math.min(5.0,p))}") }
    out.close()
    sc.stop()
  }
}
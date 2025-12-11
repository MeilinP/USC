import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.GBTRegressor

object Task2_3 {
  case class Triplet(u: String, b: String, r: Double)

  def safeD(v: Any, default: Double = 0.0): Double = {
    try {
      v match {
        case null => default
        case d: Double => d
        case f: Float => f.toDouble
        case i: Int => i.toDouble
        case l: Long => l.toDouble
        case s: String if s.trim.isEmpty || s.equalsIgnoreCase("nan") => default
        case s: String => s.toDouble
        case _ => default
      }
    } catch { case _: Throwable => default }
  }

  def main(args: Array[String]): Unit = {
    val folder = args(0)
    val valPath = args(1)
    val outPath = args(2)

    val spark = SparkSession.builder().appName("task2_3_hybrid_scala").getOrCreate()
    import spark.implicits._
    spark.sparkContext.setLogLevel("ERROR")

    val trainRaw = spark.read.textFile(folder.stripSuffix("/") + "/yelp_train.csv")
    val head = trainRaw.first()
    val train = trainRaw.filter(_ != head)
      .map(_.split(","))
      .map(a => Triplet(a(0), a(1), safeD(a(2), 3.5)))
      .rdd

    val busUsers = train.map(t => (t.b, t.u)).groupByKey().mapValues(_.toSet).collectAsMap()
    val userBus = train.map(t => (t.u, t.b)).groupByKey().mapValues(_.toSet).collectAsMap()
    val busUserRating = train.map(t => (t.b, (t.u, t.r))).groupByKey().mapValues(_.toMap).collectAsMap()
    val busAvg = train.map(t => (t.b, t.r)).groupByKey().mapValues(v => v.sum / v.size).collectAsMap()
    val userAvg = train.map(t => (t.u, t.r)).groupByKey().mapValues(v => v.sum / v.size).collectAsMap()

    val busUsersB = spark.sparkContext.broadcast(busUsers)
    val userBusB = spark.sparkContext.broadcast(userBus)
    val burB = spark.sparkContext.broadcast(busUserRating)
    val busAvgB = spark.sparkContext.broadcast(busAvg)
    val userAvgB = spark.sparkContext.broadcast(userAvg)

    val userDF = spark.read.json(folder.stripSuffix("/") + "/user.json")
      .select("user_id", "review_count", "average_stars", "useful", "funny", "cool", "fans", "elite")

    val bizDF = spark.read.json(folder.stripSuffix("/") + "/business.json")
      .select("business_id", "stars", "review_count", "is_open", "latitude", "longitude")

    val userMap = userDF.rdd.map { r =>
      val uid = r.getAs[String]("user_id")
      val elite = Option(r.getAs[String]("elite")).getOrElse("")
      val eliteYears = if (elite.trim.isEmpty || elite.equalsIgnoreCase("None")) 0.0 else elite.split(",").length.toDouble
      val rc = safeD(r.getAs[Long]("review_count"), 0)
      val avg = safeD(r.getAs[Double]("average_stars"), 3.5)
      val useful = safeD(r.getAs[Long]("useful"), 0)
      val funny = safeD(r.getAs[Long]("funny"), 0)
      val cool = safeD(r.getAs[Long]("cool"), 0)
      val fans = safeD(r.getAs[Long]("fans"), 0)
      uid -> Array[Double](rc, avg, math.log1p(useful + funny + cool), math.log1p(fans), eliteYears)
    }.collectAsMap()

    val bizMap = bizDF.rdd.map { r =>
      val bid = r.getAs[String]("business_id")
      val lat = safeD(r.getAs[Double]("latitude"), 0)
      val lon = safeD(r.getAs[Double]("longitude"), 0)
      val rc = safeD(r.getAs[Long]("review_count"), 0)
      val stars = safeD(r.getAs[Double]("stars"), 3.5)
      val isOpen = safeD(r.getAs[Long]("is_open"), 1)
      bid -> Array[Double](stars, rc, math.log1p(rc), isOpen, lat, lon)
    }.collectAsMap()

    val userB = spark.sparkContext.broadcast(userMap)
    val bizB = spark.sparkContext.broadcast(bizMap)

    val gMean = train.map(_.r).mean()
    val uStat = train.map(t => (t.u, t.r)).groupByKey().mapValues { vs =>
      val arr = vs.toArray
      val n = arr.length
      val avg = arr.sum / n
      val v = if (n > 1) arr.map(x => (x - avg) * (x - avg)).sum / n else 0.0
      (avg, n.toDouble, v)
    }.collectAsMap()
    val bStat = train.map(t => (t.b, t.r)).groupByKey().mapValues { vs =>
      val arr = vs.toArray
      val n = arr.length
      val avg = arr.sum / n
      val v = if (n > 1) arr.map(x => (x - avg) * (x - avg)).sum / n else 0.0
      (avg, n.toDouble, v)
    }.collectAsMap()

    val uStatB = spark.sparkContext.broadcast(uStat)
    val bStatB = spark.sparkContext.broadcast(bStat)
    val gB = spark.sparkContext.broadcast(gMean)

    def build(u: String, b: String): Array[Double] = {
      val uf = userB.value.getOrElse(u, Array(0.0, 3.5, 0.0, 0.0, 0.0))
      val bf = bizB.value.getOrElse(b, Array(3.5, 0.0, 0.0, 1.0, 0.0, 0.0))
      val us = uStatB.value.getOrElse(u, (3.5, 0.0, 0.0))
      val bs = bStatB.value.getOrElse(b, (3.5, 0.0, 0.0))
      val ua = us._1
      val uc = us._2
      val uv = us._3
      val ba = bs._1
      val bc = bs._2
      val bv = bs._3
      val diff = math.abs(ua - ba)
      uf ++ bf ++ Array(ua, uc, uv, ba, bc, bv, diff, ua * ba)
    }

    val trainDF = train.map(t => (t.u, t.b, Vectors.dense(build(t.u, t.b)), t.r)).toDF("uid", "bid", "features", "label")
    val gbt = new GBTRegressor().setLabelCol("label").setFeaturesCol("features").setMaxDepth(6).setMaxIter(100).setStepSize(0.1)
    val model = gbt.fit(trainDF)

    def itemBased(biz: String, user: String): Double = {
      val rated = userBusB.value.getOrElse(user, Set.empty[String])
      if (rated.isEmpty) return 3.5
      if (!busUsersB.value.contains(biz)) return userAvgB.value.getOrElse(user, 3.5)

      val sims = rated.toSeq.flatMap { b1 =>
        val common = busUsersB.value.getOrElse(biz, Set.empty[String]).intersect(busUsersB.value.getOrElse(b1, Set.empty[String]))
        val w = if (common.size <= 1) {
          (5.0 - math.abs(busAvgB.value.getOrElse(biz, 3.5) - busAvgB.value.getOrElse(b1, 3.5))) / 5.0
        } else {
          val r1 = common.toSeq.map(u => safeD(burB.value.getOrElse(biz, Map.empty).getOrElse(u, 3.5)))
          val r2 = common.toSeq.map(u => safeD(burB.value.getOrElse(b1, Map.empty).getOrElse(u, 3.5)))
          val m1 = r1.sum / r1.size
          val m2 = r2.sum / r2.size
          val a = r1.map(_ - m1)
          val b = r2.map(_ - m2)
          val num = a.indices.map(i => a(i) * b(i)).sum
          val den = math.sqrt(a.map(x => x * x).sum) * math.sqrt(b.map(x => x * x).sum)
          if (den == 0) 0.0 else num / den
        }
        val ur = burB.value.getOrElse(b1, Map.empty).get(user)
        ur.map(r => (w, r))
      }.sortBy(-_._1).take(15)

      val num = sims.map { case (w, r) => w * r }.sum
      val den = sims.map { case (w, _) => math.abs(w) }.sum
      if (den == 0.0) 3.5 else num / den
    }

    val valRaw = spark.read.textFile(valPath)
    val vhead = valRaw.first()
    val pairs = valRaw.filter(_ != vhead).map(_.split(",")).map(a => (a(0), a(1))).collect()

    val preds = pairs.map { case (u, b) =>
      val modelP = {
        val v = Vectors.dense(build(u, b))
        val df = Seq((u, b, v)).toDF("user_id", "business_id", "features")
        val p = model.transform(df).select("prediction").first().getDouble(0)
        math.max(1.0, math.min(5.0, p))
      }
      val itemP = itemBased(b, u)
      val finalP = 0.1 * itemP + 0.9 * modelP
      (u, b, math.max(1.0, math.min(5.0, finalP)))
    }.sortBy { case (u, b, _) => (u, b) }

    val headerOut = "user_id,business_id,prediction\n"
    val content = preds.map { case (u, b, p) => s"$u,$b,%.6f".format(p) }.mkString(headerOut, "\n", "\n")
    import java.nio.file.{Paths, Files}
    import java.nio.charset.StandardCharsets
    Files.write(Paths.get(outPath), content.getBytes(StandardCharsets.UTF_8))
    spark.stop()
  }
}
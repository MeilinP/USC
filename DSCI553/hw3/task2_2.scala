import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.GBTRegressor

object Task2_2 {
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

    val spark = SparkSession.builder().appName("task2_2_model_based_scala").getOrCreate()
    import spark.implicits._
    spark.sparkContext.setLogLevel("ERROR")

    val trainRaw = spark.read.textFile(folder.stripSuffix("/") + "/yelp_train.csv")
    val head = trainRaw.first()
    val train = trainRaw.filter(_ != head)
      .map(_.split(","))
      .map(a => Triplet(a(0), a(1), safeD(a(2), 3.5)))
      .rdd

    val userDF = spark.read.json(folder.stripSuffix("/") + "/user.json")
      .select("user_id", "review_count", "average_stars", "useful", "funny", "cool", "fans", "elite", "yelping_since")

    val bizDF = spark.read.json(folder.stripSuffix("/") + "/business.json")
      .select("business_id", "stars", "review_count", "is_open", "latitude", "longitude")

    val userMap = userDF.rdd.map { r =>
      val uid = r.getAs[String]("user_id")
      val elite = Option(r.getAs[String]("elite")).getOrElse("")
      val eliteYears = if (elite.trim.isEmpty || elite.equalsIgnoreCase("None")) 0.0 else elite.split(",").length.toDouble
      val since = Option(r.getAs[String]("yelping_since")).getOrElse("")
      val sinceYear = try { since.split("-")(0).toInt } catch { case _: Throwable => 2025 }
      val userAgeLog = math.log1p(math.max(0, 2025 - sinceYear))
      val rc = safeD(r.getAs[Long]("review_count"), 0)
      val avg = safeD(r.getAs[Double]("average_stars"), 3.5)
      val useful = safeD(r.getAs[Long]("useful"), 0)
      val funny = safeD(r.getAs[Long]("funny"), 0)
      val cool = safeD(r.getAs[Long]("cool"), 0)
      val fans = safeD(r.getAs[Long]("fans"), 0)
      uid -> Array[Double](
        rc,
        math.log1p(rc),
        avg,
        math.log1p(useful),
        math.log1p(funny),
        math.log1p(cool),
        math.log1p(fans),
        math.log1p(useful + funny + cool),
        eliteYears,
        userAgeLog,
        math.abs(avg - 3.5)
      )
    }.collectAsMap()

    val bizMap = bizDF.rdd.map { r =>
      val bid = r.getAs[String]("business_id")
      val lat = safeD(r.getAs[Double]("latitude"), 0)
      val lon = safeD(r.getAs[Double]("longitude"), 0)
      val rc = safeD(r.getAs[Long]("review_count"), 0)
      val stars = safeD(r.getAs[Double]("stars"), 3.5)
      val isOpen = safeD(r.getAs[Long]("is_open"), 1)
      bid -> Array[Double](
        stars,
        rc,
        math.log1p(rc),
        isOpen,
        lat,
        lon,
        lat * lon
      )
    }.collectAsMap()

    val userB = spark.sparkContext.broadcast(userMap)
    val bizB = spark.sparkContext.broadcast(bizMap)

    val gMean = train.map(_.r).mean()
    val userStats = train.map(t => (t.u, t.r)).groupByKey().mapValues { vs =>
      val arr = vs.toArray
      val n = arr.length
      val avg = arr.sum / n
      val v = if (n > 1) arr.map(x => (x - avg) * (x - avg)).sum / n else 0.0
      (avg, n.toDouble, v)
    }.collectAsMap()

    val bizStats = train.map(t => (t.b, t.r)).groupByKey().mapValues { vs =>
      val arr = vs.toArray
      val n = arr.length
      val avg = arr.sum / n
      val v = if (n > 1) arr.map(x => (x - avg) * (x - avg)).sum / n else 0.0
      (avg, n.toDouble, v)
    }.collectAsMap()

    val uStatB = spark.sparkContext.broadcast(userStats)
    val bStatB = spark.sparkContext.broadcast(bizStats)
    val gB = spark.sparkContext.broadcast(gMean)

    def build(u: String, b: String): Array[Double] = {
      val uf = userB.value.getOrElse(u, Array(0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
      val bf = bizB.value.getOrElse(b, Array(3.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
      val us = uStatB.value.getOrElse(u, (3.5, 0.0, 0.0))
      val bs = bStatB.value.getOrElse(b, (3.5, 0.0, 0.0))
      val uAvg = us._1
      val uCnt = us._2
      val uVar = us._3
      val bAvg = bs._1
      val bCnt = bs._2
      val bVar = bs._3
      val mU = 10.0
      val mB = 10.0
      val g = gB.value
      val uShr = if (uCnt + mU > 0) (uCnt * uAvg + mU * g) / (uCnt + mU) else g
      val bShr = if (bCnt + mB > 0) (bCnt * bAvg + mB * g) / (bCnt + mB) else g
      val diff = math.abs(uAvg - bAvg)
      val prod = uAvg * bAvg
      uf ++ bf ++ Array(uAvg, uCnt, uVar, bAvg, bCnt, bVar, uShr, bShr, math.log1p(uCnt), math.log1p(bCnt), uAvg - g, bAvg - g, (uAvg - g) * (bAvg - g), diff, prod)
    }

    val trainDF = train.map { t => (t.u, t.b, Vectors.dense(build(t.u, t.b)), t.r) }.toDF("uid", "bid", "features", "label")

    val gbt = new GBTRegressor().setLabelCol("label").setFeaturesCol("features").setMaxDepth(6).setMaxIter(100).setStepSize(0.1)
    val model = gbt.fit(trainDF)

    val valRaw = spark.read.textFile(valPath)
    val vhead = valRaw.first()
    val pairs = valRaw.filter(_ != vhead).map(_.split(",")).map(a => (a(0), a(1)))

    val testDF = pairs.map { case (u, b) => (u, b, Vectors.dense(build(u, b))) }.toDF("user_id", "business_id", "features")

    val preds = model.transform(testDF)
      .select("user_id", "business_id", "prediction")
      .rdd
      .map { r =>
        val u = r.getString(0)
        val b = r.getString(1)
        val p = math.max(1.0, math.min(5.0, r.getDouble(2)))
        (u, b, p)
      }
      .collect()
      .sortBy { case (u, b, _) => (u, b) }

    val headerOut = "user_id,business_id,prediction\n"
    val content = preds.map { case (u, b, p) => s"$u,$b,%.6f".format(p) }.mkString(headerOut, "\n", "\n")
    import java.nio.file.{Paths, Files}
    import java.nio.charset.StandardCharsets
    Files.write(Paths.get(outPath), content.getBytes(StandardCharsets.UTF_8))
    spark.stop()
  }
}
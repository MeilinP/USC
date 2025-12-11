import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import scala.collection.mutable
import scala.collection.mutable.{Map => MutableMap, Set => MutableSet, ArrayBuffer, Queue}

object task2 {
  
  def computeEdgeBetweenness(
    adjacencyMap: Map[String, Set[String]], 
    nodeList: List[String]
  ): List[((String, String), Double)] = {
    
    val edgeBetweennessScores = MutableMap[(String, String), Double]()
    
    for (sourceNode <- nodeList) {
      val predecessorMap = MutableMap[String, MutableSet[String]]()
      val distanceMap = MutableMap[String, Int]()
      val shortestPathCount = MutableMap[String, Double]()
      val traversalOrder = ArrayBuffer[String]()
      val bfsQueue = Queue[String]()
      
      bfsQueue.enqueue(sourceNode)
      val exploredNodes = MutableSet[String]()
      exploredNodes.add(sourceNode)
      distanceMap(sourceNode) = 0
      shortestPathCount(sourceNode) = 1.0
      
      while (bfsQueue.nonEmpty) {
        val currentNode = bfsQueue.dequeue()
        traversalOrder += currentNode
        
        for (adjacentNode <- adjacencyMap(currentNode)) {
          if (!exploredNodes.contains(adjacentNode)) {
            bfsQueue.enqueue(adjacentNode)
            exploredNodes.add(adjacentNode)
            
            if (!predecessorMap.contains(adjacentNode)) {
              predecessorMap(adjacentNode) = MutableSet[String]()
            }
            predecessorMap(adjacentNode).add(currentNode)
            
            shortestPathCount(adjacentNode) = shortestPathCount.getOrElse(adjacentNode, 0.0) + 
                                              shortestPathCount(currentNode)
            distanceMap(adjacentNode) = distanceMap(currentNode) + 1
            
          } else if (distanceMap(adjacentNode) == distanceMap(currentNode) + 1) {
            if (!predecessorMap.contains(adjacentNode)) {
              predecessorMap(adjacentNode) = MutableSet[String]()
            }
            predecessorMap(adjacentNode).add(currentNode)
            
            shortestPathCount(adjacentNode) = shortestPathCount(adjacentNode) + 
                                              shortestPathCount(currentNode)
          }
        }
      }
      
      val nodeWeightMap = MutableMap[String, Double]()
      for (node <- traversalOrder) {
        nodeWeightMap(node) = 1.0
      }
      
      val edgeContribution = MutableMap[(String, String), Double]()
      
      for (node <- traversalOrder.reverse) {
        if (predecessorMap.contains(node)) {
          for (predecessor <- predecessorMap(node)) {
            val contribution = nodeWeightMap(node) * 
                             (shortestPathCount(predecessor) / shortestPathCount(node))
            nodeWeightMap(predecessor) = nodeWeightMap(predecessor) + contribution
            
            val edgeKey = if (node < predecessor) (node, predecessor) else (predecessor, node)
            edgeContribution(edgeKey) = edgeContribution.getOrElse(edgeKey, 0.0) + contribution
          }
        }
      }
      
      for ((edge, contributionValue) <- edgeContribution) {
        edgeBetweennessScores(edge) = edgeBetweennessScores.getOrElse(edge, 0.0) + 
                                      contributionValue / 2.0
      }
    }
    
    edgeBetweennessScores.toList.sortBy { case (edge, score) => (-score, edge._1, edge._2) }
  }
  
  def findCommunities(workingGraph: Map[String, Set[String]], vertices: List[String]): List[List[String]] = {
    val visitedSet = MutableSet[String]()
    val componentList = ArrayBuffer[List[String]]()
    
    val remainingVertices = vertices.to[mutable.Queue]
    
    while (remainingVertices.nonEmpty) {
      val startVertex = remainingVertices.dequeue()
      
      if (!visitedSet.contains(startVertex)) {
        val componentQueue = Queue[String]()
        componentQueue.enqueue(startVertex)
        visitedSet.add(startVertex)
        val component = ArrayBuffer[String]()
        
        while (componentQueue.nonEmpty) {
          val vertex = componentQueue.dequeue()
          component += vertex
          
          for (neighbor <- workingGraph(vertex)) {
            if (!visitedSet.contains(neighbor)) {
              remainingVertices.dequeueAll(_ == neighbor)
              componentQueue.enqueue(neighbor)
              visitedSet.add(neighbor)
            }
          }
        }
        
        componentList += component.sorted.toList
      }
    }
    
    componentList.toList
  }
  
  def calculateModularity(
    communities: List[List[String]], 
    originalGraph: Map[String, Set[String]], 
    totalEdges: Int
  ): Double = {
    
    if (totalEdges == 0) return 0.0
    
    val nodeDegrees = originalGraph.map { case (node, neighbors) => (node, neighbors.size) }
    
    var modularityValue = 0.0
    
    for (component <- communities) {
      for (nodeI <- component) {
        for (nodeJ <- component) {
          val hasEdge = if (originalGraph(nodeI).contains(nodeJ)) 1.0 else 0.0
          modularityValue += hasEdge - 
                           (nodeDegrees(nodeI) * nodeDegrees(nodeJ)) / (2.0 * totalEdges)
        }
      }
    }
    
    modularityValue / (2.0 * totalEdges)
  }
  
  def main(args: Array[String]): Unit = {
    val thresholdValue = args(0).toInt
    val dataInputPath = args(1)
    val betweennessFilePath = args(2)
    val communityFilePath = args(3)
    
    val sc = new SparkContext("local[*]", "task2")
    sc.setLogLevel("ERROR")
    
    val dataLines = sc.textFile(dataInputPath)
    val headerLine = dataLines.first()
    val records = dataLines.filter(line => line != headerLine)
      .map(line => {
        val parts = line.split(",")
        (parts(0), parts(1))
      })
    
    val userToBusinesses = records.groupByKey().mapValues(_.toSet)
    val businessSetByUser = userToBusinesses.collectAsMap().toMap
    
    val allUsers = records.map(_._1).distinct().collect().toList
    
    val edgeList = ArrayBuffer[(String, String)]()
    val vertexCollection = MutableSet[String]()
    
    for (i <- allUsers.indices) {
      for (j <- i + 1 until allUsers.length) {
        val userA = allUsers(i)
        val userB = allUsers(j)
        
        val commonBusinesses = businessSetByUser(userA).intersect(businessSetByUser(userB))
        if (commonBusinesses.size >= thresholdValue) {
          edgeList += ((userA, userB))
          edgeList += ((userB, userA))
          vertexCollection.add(userA)
          vertexCollection.add(userB)
        }
      }
    }
    
    val vertices = vertexCollection.toList
    
    val adjacencyStructure = MutableMap[String, MutableSet[String]]()
    
    for ((userX, userY) <- edgeList) {
      if (!adjacencyStructure.contains(userX)) {
        adjacencyStructure(userX) = MutableSet[String]()
      }
      adjacencyStructure(userX).add(userY)
      
      if (!adjacencyStructure.contains(userY)) {
        adjacencyStructure(userY) = MutableSet[String]()
      }
      adjacencyStructure(userY).add(userX)
    }
    
    val adjacencyMap = adjacencyStructure.map { case (k, v) => (k, v.toSet) }.toMap
    
    var betweennessResults = computeEdgeBetweenness(adjacencyMap, vertices)
    
    val betweennessWriter = new java.io.PrintWriter(betweennessFilePath)
    try {
      for ((edgeTuple, score) <- betweennessResults) {
        betweennessWriter.println(s"$edgeTuple,${BigDecimal(score).setScale(5, BigDecimal.RoundingMode.HALF_UP)}")
      }
    } finally {
      betweennessWriter.close()
    }
    
    val workingGraphMutable = MutableMap[String, MutableSet[String]]()
    for ((node, neighbors) <- adjacencyMap) {
      workingGraphMutable(node) = neighbors.to[MutableSet]
    }
    
    val totalEdges = betweennessResults.length
    val nodeDegrees = adjacencyMap.map { case (node, neighbors) => (node, neighbors.size) }
    
    var bestModularityScore = Double.NegativeInfinity
    var optimalCommunities = List[List[String]]()
    
    while (betweennessResults.nonEmpty) {
      val workingGraphImmutable = workingGraphMutable.map { case (k, v) => (k, v.toSet) }.toMap
      val componentList = findCommunities(workingGraphImmutable, vertices)
      
      val modularityValue = calculateModularity(componentList, adjacencyMap, totalEdges)
      
      if (modularityValue > bestModularityScore) {
        bestModularityScore = modularityValue
        optimalCommunities = componentList
      }
      
      val maxBetweenness = betweennessResults.head._2
      for ((edge, betweennessVal) <- betweennessResults) {
        if (betweennessVal >= maxBetweenness) {
          workingGraphMutable(edge._1).remove(edge._2)
          workingGraphMutable(edge._2).remove(edge._1)
        }
      }
      
      val workingGraphForBetweenness = workingGraphMutable.map { case (k, v) => (k, v.toSet) }.toMap
      betweennessResults = computeEdgeBetweenness(workingGraphForBetweenness, vertices)
    }
    
    val sortedCommunities = optimalCommunities.sortBy(c => (c.size, c.head))
    s
    val communityWriter = new java.io.PrintWriter(communityFilePath)
    try {
      for (community <- sortedCommunities) {
        communityWriter.println(community.map(u => s"'$u'").mkString(", "))
      }
    } finally {
      communityWriter.close()
    }
    
    sc.stop()
  }
}
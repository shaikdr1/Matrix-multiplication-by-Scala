
val M = Array(Array(1.0, 2.0), Array(3.0, 4.0))
val N = Array(Array(5.0, 6.0), Array(7.0, 8.0))

val M_RDD_Small = sc.parallelize(M.zipWithIndex.flatMap { case (row, i) => row.zipWithIndex.map { case (value, j) => ((i.toInt, j.toInt), value) } })
val N_RDD_Small = sc.parallelize(N.zipWithIndex.flatMap { case (row, i) => row.zipWithIndex.map { case (value, j) => ((i.toInt, j.toInt), value) } })


def COOMatrixMultiply(M: RDD[((Int, Int), Double)], N: RDD[((Int, Int), Double)]): RDD[((Int, Int), Double)] = {
  val rowMatrix = M.map { case ((i, k), valuem) => ( k, (i, valuem)) }
  val colMatrix = N.map { case ((k, j), valuen) => ( k, (j, valuen)) }
  val result = rowMatrix.join(colMatrix).map{case(_, ((i, valuem),(j, valuen)))=>((i,j),valuem*valuen)}
               .reduceByKey(_+_)
               result
 
}


val R_RDD_Small = COOMatrixMultiply(M_RDD_Small, N_RDD_Small)
R_RDD_Small.collect.foreach(println)


def manualMatrixMultiply(M: Array[Array[Double]], N: Array[Array[Double]]): Array[Array[Double]] = {
  val result = Array.ofDim[Double](M.length, N(0).length)
  for {
    i <- M.indices
    j <- N(0).indices
    k <- M(0).indices
  } result(i)(j) += M(i)(k) * N(k)(j)
  result
}

val result_manual = R_RDD_Small.collect().map { case ((i, j), value) => (i, j, value) }.sortBy { case (i, j, _) => (i, j) }

val resultArray = Array.ofDim[Double](M.length, N(0).length)

for ((i, j, value) <- result_manual) {
  resultArray(i)(j) = value
}

val expectedResult = manualMatrixMultiply(M, N)
assert(resultArray.deep == expectedResult.deep, "Result mismatch")


import scala.util.Random

// Function to generate random coordinate matrices for large datasets
def randomCOOMatrix ( n: Int, m: Int ): RDD[((Int,Int),Double)] = {
  val max = 10
  val l = Random.shuffle((0 until n).toList)
  val r = Random.shuffle((0 until m).toList)
  sc.parallelize(l)
    .flatMap{ i => val rand = new Random()
              r.map{ j => ((i.toInt,j.toInt),rand.nextDouble()*max) } 
              }
}


val n = 1024
val m = 1024
val M_RDD_Large = randomCOOMatrix(n,m)
val N_RDD_Large = randomCOOMatrix(m,n)

val R_RDD_Large = COOMatrixMultiply(M_RDD_Large, N_RDD_Large)

R_RDD_Large.count



import org.apache.spark.mllib.linalg.distributed._


val M_Block_Matrix = new CoordinateMatrix(M_RDD_Small.map { case ((i, j), value) => MatrixEntry(i, j, value)}).toBlockMatrix()
val N_Block_Matrix = new CoordinateMatrix(N_RDD_Small.map { case ((i, j), value) => MatrixEntry(i, j, value)}).toBlockMatrix()



M_Block_Matrix.blocks.collect.foreach(println)



N_Block_Matrix.blocks.collect.foreach(println)



val R_Block_Small = M_Block_Matrix.multiply(N_Block_Matrix)


R_Block_Small.blocks.collect.foreach(println)

val r_b = 64
val c_b = 64




val M_Block_Matrix_Large = new CoordinateMatrix(M_RDD_Large.map { case ((i, j), value) => MatrixEntry(i, j, value)}).toBlockMatrix(r_b,c_b)
val N_Block_Matrix_Large = new CoordinateMatrix(N_RDD_Large.map { case ((i, j), value) => MatrixEntry(i, j, value)}).toBlockMatrix(c_b,c_b)



M_Block_Matrix_Large.validate
N_Block_Matrix_Large.validate


assert(M_Block_Matrix_Large.numRowBlocks == 16, "Result mismatch")
assert(M_Block_Matrix_Large.numColBlocks == 16, "Result mismatch")


assert(N_Block_Matrix_Large.numRowBlocks == 16, "Result mismatch")
assert(N_Block_Matrix_Large.numColBlocks == 16, "Result mismatch")


val R_Block_Large = M_Block_Matrix_Large.multiply(N_Block_Matrix_Large)

R_Block_Large.blocks.count

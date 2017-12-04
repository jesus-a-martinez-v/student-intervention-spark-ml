import java.io.File

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.udf

object StudentIntervention extends App {
  implicit val spark: SparkSession = getSparkSession
  import spark.sqlContext.implicits._

  val data = getData(new File(this.getClass.getClassLoader.getResource("student-data.csv").toURI).getPath)
  data.printSchema()

  val numberOfRows = data.count()
  val numberOfColumns = data.columns.length
  val numberOfFeatures = numberOfColumns - 1

  println(s"Number of rows: $numberOfRows")
  println(s"Number of columns: $numberOfColumns")
  println(s"Number of features: $numberOfFeatures")

  val numberOfTotalStudents = data.count()
  val numberOfStudentsThatPassed = data.select("passed").where("passed = 'yes'").count()
  val numberOfStudentsThatFailed = numberOfTotalStudents - numberOfStudentsThatPassed
  val graduationRate = 100.0 * (numberOfStudentsThatPassed.toDouble / numberOfTotalStudents.toDouble)

  println(s"Number of students: $numberOfTotalStudents")
  println(s"Number of students that passed: $numberOfStudentsThatPassed")
  println(s"Number of students that failed: $numberOfStudentsThatFailed")
  println(f"Graduation rate: $graduationRate%1.2f" + "%")

  val yesNoToBinary = udf((value: String) => value match {
    case "yes" => 1
    case "no" => 0
  })

  val categoricalFeatureNames = List("school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian")
  val yesOrNoFeatureNames = List("schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic", "passed")

  val preprocessedData = yesOrNoFeatureNames.foldLeft(data)((df, c) => df.withColumn(c, yesNoToBinary.apply(df(c))))

  val featureColumnsNames = preprocessedData.columns.toList.filterNot(_ == "passed")
  val featureColumns = featureColumnsNames.map(preprocessedData.apply)

  val labelColumn = preprocessedData("passed").as("label")

  val allColumns = labelColumn :: featureColumns

  val dataWithLabel = preprocessedData.select(allColumns:_*)

  val Array(trainingData, testData) = dataWithLabel.randomSplit(Array(0.75, 0.25), seed = 42)

  val assembler = new VectorAssembler()
    .setInputCols(featureColumnsNames.map {c => if (categoricalFeatureNames.contains(c)) s"${c}Vec" else c} toArray)
    .setOutputCol("features")

  val classifier = new RandomForestClassifier()

  val paramGrid = new ParamGridBuilder()
    .addGrid(classifier.impurity, Array("gini", "entropy"))
    .addGrid(classifier.numTrees, Array(10, 25, 40, 50, 100, 200))
    .addGrid(classifier.minInstancesPerNode, Array(2, 3, 5, 10))
    .build()

  val pipeline = new Pipeline().setStages((categoricalFeatureNames.map(getStringIndexer) ::: categoricalFeatureNames.map(getOneHotEncoder) ::: List(assembler, classifier)).toArray)

  val trainValidationSplit = new TrainValidationSplit()
    .setEstimator(pipeline)
    .setEvaluator(new BinaryClassificationEvaluator())
    .setEstimatorParamMaps(paramGrid)
    .setTrainRatio(0.8)

  val model = trainValidationSplit.fit(trainingData)
  val results = model.transform(testData)

  val predictionAndLabels = results.select(results("prediction"), results("label")).as[(Double, Double)].rdd
  val metrics = new MulticlassMetrics(predictionAndLabels)
  val fScore = metrics.fMeasure(1.0)
  println(fScore)

  spark.stop()

  def getSparkSession = SparkSession
    .builder()
    .master("local[*]")
    .appName("StudentIntervention")
    .getOrCreate()

  def getData(path: String) =
    spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .format("csv")
      .load(path)

  def getStringIndexer(columnName: String) = new StringIndexer()
    .setInputCol(columnName)
    .setOutputCol(s"${columnName}Index")

  def getOneHotEncoder(columnName: String) = new OneHotEncoder()
    .setInputCol(s"${columnName}Index")
    .setOutputCol(s"${columnName}Vec")

}

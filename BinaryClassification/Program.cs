using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;
using MongoDB.Bson;
using MongoDB.Driver;
using static BinaryClassification.TrabajoPlanificadoData;

//Based on https://docs.microsoft.com/es-es/dotnet/machine-learning/tutorials/sentiment-analysis
namespace BinaryClassification
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static string mongoURI = "mongodb://energy:1980abc$1@NUEVOEZALOR:27017/Energy";
        static readonly MongoClient client = new MongoClient(mongoURI);

        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            MLContext mlContext = new MLContext();
            TrainCatalogBase.TrainTestData splitDataView = LoadData(mlContext);
            ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);
            Evaluate(mlContext, model, splitDataView.TestSet);
            UseModelWithSingleItem(mlContext, model);

            Console.WriteLine("Goodbye World!");
            var end = Console.ReadLine();


        }
        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {
            //var pipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: DefaultColumnNames.Features, inputColumnName: nameof(SentimentData.SentimentText))
            //    .Append(mlContext.BinaryClassification.Trainers.FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeaves: 20));
            /*
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Valorable")
                .Append(mlContext.Transforms.Concatenate("Features", "PrevisionPreciosPorFechaHora", "CostesOperacionPorFechaHora", "PrevisionProduccionPorFechaHora", "RetribucionesPorFechaHora"))
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.BinaryClassification.Trainers.StochasticDualCoordinateAscent(labelColumnName: "Valorable", featureColumnName: "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
                */
            var pipeline = mlContext.Transforms.Concatenate("Features", "PrevisionPreciosPorFechaHora", "CostesOperacionPorFechaHora", "PrevisionProduccionPorFechaHora", "RetribucionesPorFechaHora")
                .Append(mlContext.BinaryClassification.Trainers.FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeaves: 20));
            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = pipeline.Fit(splitTrainSet);

            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();
            return model;
        }

        public static TrainCatalogBase.TrainTestData LoadData(MLContext mlContext)
        {
            //IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);
            //TrainCatalogBase.TrainTestData splitDataView = mlContext.BinaryClassification.TrainTestSplit(dataView, testFraction: 0.2);

            var database = client.GetDatabase("Energy");
            //var collectionTP = database.GetCollection<BsonDocument>("Resultado.TrabajosPlanificadosPropuestas");
            var collection = database.GetCollection<TrabajoPlanificadoMongo>("TrabajoPlanificado");
            var documents = collection.Find<TrabajoPlanificadoMongo>(new BsonDocument()).ToList();

            var TPdata = new List<TrabajoPlanificadoPropuestaData>();
            int vcount = 0;
            //TrabajoPlanificado
            foreach (var document in documents)
            {
                var rowSet = new List<TrabajoPlanificadoPropuestaData>();

                var row = new TrabajoPlanificadoPropuestaData();
                bool hasValorable = false;
                //Resultado.Propuestas
                foreach (var propuesta in document.Resultado.TrabajosPlanificadosPropuestas)
                {
                    //Horas por Propuesta
                    for (DateTime i = propuesta.FechaHoraInicio; i < propuesta.FechaHoraFin; i = i.AddHours(1))
                    {
                        decimal value = 0;
                        if (document.Resultado.PrevisionPreciosPorFechaHora.TryGetValue(i, out value))
                            row.PrevisionPreciosPorFechaHora += (float)value;
                        else
                            row.PrevisionPreciosPorFechaHora += (float)value;

                        document.Resultado.CostesOperacionPorFechaHora.TryGetValue(i, out value);
                        row.CostesOperacionPorFechaHora += (float)value;

                        document.Resultado.PrevisionProduccionPorFechaHora.TryGetValue(i, out value);
                        row.PrevisionProduccionPorFechaHora += (float)value;

                        document.Resultado.RetribucionesPorFechaHora.TryGetValue(i, out value);
                        row.RetribucionesPorFechaHora += (float)value;
                    }
                    row.Valorable = propuesta.Valorable;
                    if (propuesta.Valorable)
                    {
                        hasValorable = true;
                        vcount++;
                    }
                    rowSet.Add(row);

                }
                //Idea: Dont add propuestas that have time ranges out of the range of the measurements
                //Idea: Only add propuestas that have a valorable member in their result set.
                //Idea: because each instalation has different predictors available, train a different model per instalation
                if (hasValorable)
                {
                    TPdata = TPdata.Concat(rowSet).ToList();
                }
            }

            IDataView dataView = mlContext.Data.LoadFromEnumerable(TPdata);
            Console.WriteLine("PropuestaSets with valorable menbers: {0}", TPdata.Count());
            Console.WriteLine("Propuestas of valorable label: {0}", vcount);



            TrainCatalogBase.TrainTestData splitDataView = mlContext.BinaryClassification.TrainTestSplit(dataView, testFraction: 0.25);


            return splitDataView;
        }

        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            IDataView predictions = model.Transform(splitTestSet);
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
            SaveModelAsFile(mlContext, model);

        }
        private static void SaveModelAsFile(MLContext mlContext, ITransformer model)
        {
            using (var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(model, fs);
            Console.WriteLine("The model is saved to {0}", _modelPath);
        }
        private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        {
            //PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = model.CreatePredictionEngine<SentimentData, SentimentPrediction>(mlContext);
            PredictionEngine<TrabajoPlanificadoPropuestaData, TrabajoPlanificadoPrediction> predictionFunction = model.CreatePredictionEngine<TrabajoPlanificadoPropuestaData, TrabajoPlanificadoPrediction>(mlContext);
            TrabajoPlanificadoPropuestaData sampleStatement = new TrabajoPlanificadoPropuestaData();
            var resultprediction = predictionFunction.Predict(sampleStatement);
            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

            Console.WriteLine();
            //Console.WriteLine($"Sentiment: {sampleStatement.SentimentText} | Prediction: {(Convert.ToBoolean(resultprediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultprediction.Probability} ");

            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }
        public static void UseLoadedModelWithBatchItems(MLContext mlContext)
        {

        }
    }
}

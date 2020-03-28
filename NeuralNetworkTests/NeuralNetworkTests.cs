using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Tests
{
    [TestClass()]
    public class NeuralNetworkTests
    {
        [TestMethod()]
        public void FeedForwardTest()
        {
            var outputs = new double[] {0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1};
            var inputs = new double[,]
            {
                // Результат - Пациент болен - 1
                //             Пациент Здоров - 0

                // Неправильная температура T
                // Хороший возраст A
                // Курит S
                // Правильно питается F
                //T  A  S  F
                {1, 0, 0, 0, 0 },
                {1, 0, 0, 0, 1 },
                {1, 0, 0, 1, 0 },
                {1, 0, 0, 1, 1 },
                {1, 0, 1, 0, 0 },
                {1, 0, 1, 0, 1 },
                {1, 0, 1, 1, 0 },
                {1, 0, 1, 1, 1 },
                {1, 1, 0, 0, 0 },
                {1, 1, 0, 0, 1 },
                {1, 1, 0, 1, 0 },
                {1, 1, 0, 1, 1 },
                {1, 1, 1, 0, 0 },
                {1, 1, 1, 0, 1 },
                {1, 1, 1, 1, 0 },
                {1, 1, 1, 1, 1 }
            };



            Topology topology = new Topology(5, 1, 0.1, 10, 5);
            NeuralNetwork neuralNetwork = new NeuralNetwork(topology);
            var difference = neuralNetwork.Learn(outputs, inputs, 150000);

            var results = new List<double>();
            for (int i = 0; i < outputs.Length; i++)
            {
                var row = NeuralNetwork.GetRow(inputs, i);
                var res = neuralNetwork.Predict(row).Output;
                results.Add(res);
            }

            for (int i = 0; i < results.Count; i++)
            {
                var expected = Math.Round(outputs[i], 2);
                var actual = Math.Round(results[i], 2);
                Assert.AreEqual(expected, actual);
            }
        }

        [TestMethod()]
        public void RecognizeImage()
        {
            var parasitizedPath = @"D:\Parasitized";
            var unparasitizedPath = @"D:\Uninfected";

            var converter = new PictureConverter();
            var testParasitizedImagesInput =
                converter.Convert(@"D:\Users\Дмитри\Downloads\NeuralNetwork\NeuralNetworkTests\Images\Parasitized.png");
            var testUnparasitizedImagesInput =
                converter.Convert(@"D:\Users\Дмитри\Downloads\NeuralNetwork\NeuralNetworkTests\Images\Unparasitized.png");

            var topology = new Topology(testParasitizedImagesInput.Count, 1, 0.1, testParasitizedImagesInput.Count / 2);
            var neuralNetwork = new NeuralNetwork(topology);

            var size = 1000;
            //Обучение
            double[,] parasitizedInputs = GetData(parasitizedPath, converter, testParasitizedImagesInput, size);
            neuralNetwork.Learn(new double[] { 1 }, parasitizedInputs, 1);

            double[,] unparasitizedInputs = GetData(unparasitizedPath, converter, testParasitizedImagesInput, size);
            neuralNetwork.Learn(new double[] { 0 }, unparasitizedInputs, 1);

            var par = neuralNetwork.Predict(testParasitizedImagesInput.Select(t => (double)t).ToArray());
            var unpar = neuralNetwork.Predict(testUnparasitizedImagesInput.Select(t => (double)t).ToArray());
            Assert.AreEqual(1, Math.Round(par.Output, 2));
            Assert.AreEqual(0, Math.Round(unpar.Output, 2));
        }

        private static double[,] GetData(string parasitizedPath, PictureConverter converter, List<int> testImagesInput, int size)
        {
            var images = new double[size, testImagesInput.Count];
            var result = Directory.GetFiles(parasitizedPath);
            for (int i = 0; i < size; i++)
            {
                var image = converter.Convert(result[i]);
                for (int j = 0; j < image.Count; j++)
                {
                    images[i, j] = image[j];
                }
            }

            return images;
        }
        //[TestMethod()]
        //public void DatasetTest()
        //{
        //    var outputs = new List<double>();
        //    var inputs = new List<double[]>();
        //    using (var sr = new StreamReader("heart.csv"))
        //    {
        //        var header = sr.ReadLine();
        //        while (!sr.EndOfStream)
        //        {
        //            var row = sr.ReadLine();
        //            var values = row.Split(',').Select(v => Convert.ToDouble(v.Replace(".", ","))).ToList();
        //            var output = values.Last();
        //            var input = values.Take(values.Count - 1).ToArray();

        //            outputs.Add(output);
        //            inputs.Add(input);
        //        }
        //    }

        //    var inputSignals = new double[inputs.Count, inputs[0].Length];
        //    for (int i = 0; i < inputSignals.GetLength(0); i++)
        //    {
        //        for (int j = 0; j < inputSignals.GetLength(1); j++)
        //        {
        //            inputSignals[i, j] = inputs[i][j];
        //        }
        //    }

        //    Topology topology = new Topology(outputs.Count, 1, 0.1, 10, outputs.Count / 2);
        //    NeuralNetwork neuralNetwork = new NeuralNetwork(topology);
        //    var difference = neuralNetwork.Learn(outputs.ToArray(), inputSignals, 160000);

        //    var results = new List<double>();
        //    for (int i = 0; i < outputs.Count; i++)
        //    {
        //        var res = neuralNetwork.Predict(inputs[i]).Output;
        //        results.Add(res);
        //    }
        //    for (int i = 0; i < results.Count; i++)
        //    {
        //        var expected = Math.Round(outputs[i], 2);
        //        var actual = Math.Round(results[i], 2);
        //        Assert.AreEqual(expected, actual);
        //    }
        //}
    }
   
}
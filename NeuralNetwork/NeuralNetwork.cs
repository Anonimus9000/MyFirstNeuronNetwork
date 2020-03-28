using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        public Layer[] Layers { get; }
        public Topology Topology { get; }

        public NeuralNetwork(Topology topology)
        {
            Topology = topology;

            Layers = new Layer[Topology.TotalLayersCount];

            CreateInputLayer();
            CreateHiddenLayers();
            CreateOutputLayer();
        }

        public Neuron Predict(params double[] inputSignals)
        {
            SendSignalsToInputNeurons(inputSignals);
            FeedForwarAllLayersAfterInput();

            if (Topology.OutputCount == 1)
            {


                if (Double.IsNaN(Layers[Topology.TotalLayersCount - 1].Neurons[0].Output))
                {
                    bool b = true;
                }



                return Layers[Topology.TotalLayersCount - 1].Neurons[0];
            }
            else
            {
                return Layers[Topology.TotalLayersCount - 1].Neurons.OrderByDescending(n => n.Output).First();
            }
        }

        public double Learn(double[] expected, double[,] inputs, int epoch)
        {
            var error = 0.0;
            for (int i = 0; i < epoch; i++)
            {
                for (int j = 0; j < expected.Length; j++)
                {
                    var output = expected[j];
                    var input = GetRow(inputs, j);

                    error += BackPropagation(output, input);
                }
            }

            var result = error / epoch;
            return result;
        }

        public static double[] GetRow(double[,] matrix, int row)
        {
            var columns = matrix.GetLength(1);
            var array = new double[columns];
            for (int i = 0; i < columns; ++i)
                array[i] = matrix[row, i];
            return array;
        }

        //private double BackPropagation(double expected, params double[] inputs)
        //{
        //    double actual = Predict(inputs).Output;

        //    var difference = actual - expected;

        //    foreach (var neuron in Layers.Last().Neurons)
        //    {
        //        neuron.Learn(difference, Topology.LearningRate);
        //    }

        //    for (int j = Layers.Length - 2; j >= 0; j--)
        //    {
        //        var layer = Layers[j];
        //        var previousLayer = Layers[j + 1];

        //        for (int i = 0; i < layer.NeuronsCountInLayer; i++)
        //        {
        //            var neuron = layer.Neurons[i];

        //            for (int k = 0; k < previousLayer.NeuronsCountInLayer; k++)
        //            {
        //                var previpusNeuron = previousLayer.Neurons[k];
        //                var error = previpusNeuron.Weights[i] * previpusNeuron.Delta;
        //                neuron.Learn(error, Topology.LearningRate);
        //            }
        //        }
        //    }
        //    var result = Math.Pow(difference, 2);
        //    return result;
        //}

        private double BackPropagation(double expected, params double[] inputs)
        {
            double actual = Predict(inputs).Output;

            var result = FindingMistake(expected, actual);
            Learning();
            return result;
        }

        public void Learning()
        {
            for (int i = 1; i < Layers.Length; i++)
            {
                var layer = Layers[i];

                for (int j = 0; j < layer.NeuronsCountInLayer; j++)
                {
                    var neuron = layer.Neurons[j];
                    neuron.Learn(neuron.Mistake, Topology.LearningRate);
                }
            }
        }

        public double FindingMistake(double expected, double actual)
        {
            var outputNeuron = Layers.Last().Neurons[0];
            double DxOutput = outputNeuron.SigmoidDx(outputNeuron.Output);
            double difference = (expected - actual) * DxOutput;
            outputNeuron.FindDeltas(difference, Topology.LearningRate);

            for (int j = Layers.Length - 2; j >= 0; j--)
            {
                var layer = Layers[j];
                var previousLayer = Layers[j + 1];

                for (int i = 0; i < layer.NeuronsCountInLayer; i++)
                {
                    var neuron = layer.Neurons[i];
                    double error = 0.0;

                    for (int k = 0; k < previousLayer.NeuronsCountInLayer; k++)
                    {
                        var previpusNeuron = previousLayer.Neurons[k];
                        error += previpusNeuron.Mistake * previpusNeuron.Weights[i];

                        if (Double.IsNaN(error))
                        {
                            bool b = true;
                        }

                    }

                    DxOutput = neuron.SigmoidDx(neuron.Output);
                    difference = error * DxOutput;
                    neuron.Mistake = error;
                    neuron.FindDeltas(difference, Topology.LearningRate);
                }
            }

            var result = Math.Pow((actual - expected), 2);
            return result;

            //var difference = actual - expected;
            //Layers.Last().Neurons[0].Mistake = difference;
            ////Layers.Last().Neurons[0].Learn(difference, Topology.LearningRate);

            //for (int j = Layers.Length - 2; j >= 0; j--)
            //{
            //    var layer = Layers[j];
            //    var previousLayer = Layers[j + 1];

            //    for (int i = 0; i < layer.NeuronsCountInLayer; i++)
            //    {
            //        var neuron = layer.Neurons[i];
            //        double error = 0.0;

            //        for (int k = 0; k < previousLayer.NeuronsCountInLayer; k++)
            //        {
            //            var previpusNeuron = previousLayer.Neurons[k];                        
            //            error += previpusNeuron.Mistake * previpusNeuron.Weights[i];

            //            if (Double.IsNaN(error))
            //            {
            //                bool b = true;
            //            }

            //        }
            //        neuron.Mistake = error;
            //        //neuron.Learn(error, Topology.LearningRate);
            //    }
            //}

            ////ОБУЧЕНИЕ начало
            //for (int i = 1; i < Layers.Length; i++)
            //{
            //    var layer = Layers[i];

            //    for (int j = 0; j < layer.NeuronsCountInLayer; j++)
            //    {
            //        var neuron = layer.Neurons[j];
            //        neuron.Learn(neuron.Mistake, Topology.LearningRate);
            //    }
            //}
            ////ОБУЧЕНИЕ конец

        }

        private void FeedForwarAllLayersAfterInput()
        {
            for (int i = 1; i < Layers.Length; i++)
            {
                var previousLayerSignals = Layers[i - 1].GetSignals();
                var layer = Layers[i];

                foreach (var neuron in layer.Neurons)
                {
                    neuron.FeedForward(previousLayerSignals);
                }
            }
        }

        private void SendSignalsToInputNeurons(double[] inputSignals)
        {
            for (int i = 0; i < inputSignals.Length; i++)
            {
                var signal = new double[] { inputSignals[i] };
                var neuron = Layers[0].Neurons[i];

                neuron.FeedForward(signal);
            }
        }

        private void CreateOutputLayer()
        {
            var outputNeurons = new Neuron[Topology.OutputCount];
            var preLastLayer = Layers[Topology.TotalLayersCount - 2];

            for (int i = 0; i < Topology.OutputCount; i++)
            {
                var neuron = new OutputNeuron(preLastLayer.NeuronsCountInLayer, NeuronType.Output);
                outputNeurons[i] = neuron;
            }
            var outputLayer = new Layer(outputNeurons, NeuronType.Output);
            Layers[Topology.TotalLayersCount - 1] = outputLayer;
        }

        private void CreateHiddenLayers()
        {
            for (int j = 0; j < Topology.HiddenLayersCount; j++)
            {
                var hiddenNeuron = new Neuron[Topology.HiddenLayers[j]];
                var preLastLayer = Layers[j];

                for (int i = 0; i < Topology.HiddenLayers[j]; i++)
                {
                    var neuron = new HiddenNeuron(preLastLayer.NeuronsCountInLayer);
                    hiddenNeuron[i] = neuron;
                    Thread.Sleep(10);
                }
                var hiddenLayer = new Layer(hiddenNeuron);
                Layers[j + 1] = hiddenLayer;
            }
        }

        private void CreateInputLayer()
        {
            var inputNeurons = new Neuron[Topology.InputCount];
            for (int i = 0; i < Topology.InputCount; i++)
            {
                var neuron = new InputNeuron(1, NeuronType.Input);
                inputNeurons[i] = neuron;
            }
            var inputLayer = new Layer(inputNeurons, NeuronType.Input);
            Layers[0] = inputLayer;
        }
    }
}

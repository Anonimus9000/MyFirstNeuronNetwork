

using System;

namespace NeuralNetwork
{
    public abstract class Neuron
    {
        public double[] Weights { get; }

        public double[] Inputs { get; }

        public NeuronType NeuronType { get; }

        public double Output { get; private set; }

        public double Delta { get; private set; }

        public double[] Deltas { get; private set; }

        public double Mistake { get; set; }

        public Neuron(int inputCount, NeuronType type = NeuronType.Normal)
        {
            NeuronType = type;
            Weights = new double[inputCount];
            Inputs = new double[inputCount];
            Deltas = new double[inputCount];

            InitWeightsRandomValues(inputCount);
        }

        private void InitWeightsRandomValues(int inputCount)
        {
            var rnd = new Random();
            for (int i = 0; i < inputCount; i++)
            {
                Weights[i] = rnd.NextDouble();
                Inputs[i] = 0.0;
            }
        }

        

        public void Learn(double error, double learningRate)
        {
            if (NeuronType == NeuronType.Input)
            {
                return;
            }

            for (int i = 0; i < Weights.Length; i++)
            {
                var weight = Weights[i];
                var delta = Deltas[i];
                var newWeight = weight + delta;
                Weights[i] = newWeight;
            }
        }


        public void FindDeltas(double error, double learningRate)
        {
            
            Mistake = error;
            if (NeuronType == NeuronType.Input)
            {
                return;
            }

            for (int i = 0; i < Deltas.Length; i++)//массив смещений весов
            {
                Deltas[i] = Inputs[i] * Mistake * learningRate;
            }

        }

        public double FeedForward(double[] inputs)
        {
            //TODO выполнить проверку совпадения количества входных сигналов с количеством весов
            var sum = 0.0;
            for (int i = 0; i < inputs.Length; i++)
            {
                Inputs[i] = inputs[i];
                //if (!double.IsNaN(inputs[i] * Weights[i]) )
                //{
                //    sum += inputs[i] * Weights[i];
                //}
                if (double.IsNaN(sum))
                {
                    sum = 0.0;
                }
                sum += inputs[i] * Weights[i];
            }
            if (NeuronType != NeuronType.Input)
            {
                Output = Sigmoid(sum);
            }
            else
            {
                Output = sum;
            }
            return Output;
        }

        private double Sigmoid(double x)
        {
            double pow = Math.Exp(-x);
            double result = 1.0 / (1.0 + pow);
            return result;
        }

        public double SigmoidDx(double x)
        {
            double result = x * (1.0 - x);
            return result;
        }

        public override string ToString()
        {
            return Output.ToString();
        }
    }
}

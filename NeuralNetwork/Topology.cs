
namespace NeuralNetwork
{
    public class Topology
    {
        public int InputCount { get; }

        public int OutputCount { get; }

        public double LearningRate { get; }

        public int[] HiddenLayers { get; }

        public int HiddenLayersCount { get; }

        public int TotalLayersCount { get; }

        public Topology(int inputCount, int outputCount, double learningRate, params int[] hiddenLayers)
        {
            InputCount = inputCount;
            OutputCount = outputCount;
            LearningRate = learningRate;
            if (hiddenLayers == null)
            {
                HiddenLayers = new int[1];
                HiddenLayers[0] = inputCount / 2;
                HiddenLayersCount = HiddenLayers.Length;
            }
            else
            {
                HiddenLayers = hiddenLayers;
                HiddenLayersCount = HiddenLayers.Length;
            }
            TotalLayersCount = HiddenLayers.Length + 2;
        }
    }
}

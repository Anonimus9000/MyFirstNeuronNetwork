

namespace NeuralNetwork
{
    public class Layer
    {
        public Neuron[] Neurons { get; }

        public int NeuronsCountInLayer => Neurons?.Length ?? 0;

        public NeuronType Type;

        public Layer(Neuron[] neurons, NeuronType type = NeuronType.Normal)
        {
            //TODO: проверить все входные нейроны на соответствие типу
            Type = type;
            Neurons = neurons;
        }

        public double[] GetSignals()
        {
            var result = new double[NeuronsCountInLayer];

            for (int i = 0; i < NeuronsCountInLayer; i++)
            {
                result[i] = Neurons[i].Output;
            }
            return result;
        }

        public override string ToString()
        {
            return Type.ToString();
        }
    }
}

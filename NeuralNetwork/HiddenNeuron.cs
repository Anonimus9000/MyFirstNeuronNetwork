

namespace NeuralNetwork
{
    class HiddenNeuron : Neuron
    {
        public HiddenNeuron(int inputCount, NeuronType type = NeuronType.Normal)
            : base(inputCount, type) { }
    }
}

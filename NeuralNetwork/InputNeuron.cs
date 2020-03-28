
namespace NeuralNetwork
{
    class InputNeuron : Neuron
    {
        public InputNeuron(int inputCount, NeuronType type = NeuronType.Input)
            : base(inputCount, type) { Weights[0] = 1; }
    }
}

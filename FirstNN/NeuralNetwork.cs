using System;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace FirstNN
{
    public class NeuralNetwork
    {
        private readonly int _inputLayerSize;
        private readonly int _outputLayaerSize;
        private readonly int _hiddenLayerSize;

        public NeuralNetwork(int inputLayerSize, int outputLayaerSize, int hiddenLayerSize)
        {
            _inputLayerSize = inputLayerSize;
            _outputLayaerSize = outputLayaerSize;
            _hiddenLayerSize = hiddenLayerSize;
        }

        public Matrix<double> Forward(Matrix<double> x)
        {
            var w1 = Matrix<double>.Build.Random(_inputLayerSize, _hiddenLayerSize);
            Print("w1", w1);
            var w2 = Matrix<double>.Build.Random(_hiddenLayerSize, _outputLayaerSize);
            Print("w2", w2);

            var z2 = x * w1;
            Print("z2", z2);

            var a2 = NeuralNetwork.Sigmoid(z2);
            Print("a2", a2);

            var z3 = a2 * w2;
            Print("z3", z3);

            var yHat = NeuralNetwork.Sigmoid(z3);
            Print("yHat", yHat);

            return yHat;
        }

        private static Matrix<double> Sigmoid(Matrix<double> m)
        {
            return m.Map(SpecialFunctions.Logistic);
        }

        private static void Print(string name, Matrix<double> m)
        {
            Console.Write($"{name}: ");
            Console.WriteLine(m.ToString());
        }
    }
}
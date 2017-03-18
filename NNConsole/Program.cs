using System;
using FirstNN;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace NNConsole
{
    class Program
    {
        private const int InputLayerSize = 2;
        private const int OutputLayaerSize = 1;
        private const int HiddenLayerSize = 3;

        static void Main(string[] args)
        {
            var x = Matrix<double>.Build.DenseOfArray(new double[,]
            {
                {3, 5},
                {5, 1},
                {10, 2}
            });

            var nn = new NeuralNetwork(InputLayerSize, OutputLayaerSize, HiddenLayerSize);
            var yHat = nn.Forward(x);

            var y = Matrix<double>.Build.DenseOfArray(new double[,] { { 75 }, { 82 }, { 93 } });

            Console.Write("Press a key...");
            Console.ReadKey();
        }

        private static void Print(string name, Matrix<double> m)
        {
            Console.Write($"{name}: ");
            Console.WriteLine(m.ToString());
        }
    }
}

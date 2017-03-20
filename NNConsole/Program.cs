using System;
using FirstNN;
using MathNet.Numerics.LinearAlgebra;

namespace NNConsole
{
    class Program
    {
        private const int InputLayerSize = 2;
        private const int OutputLayerSize = 1;
        private const int HiddenLayerSize = 3;

        static void Main(string[] args)
        {
            //TrySomething();

            var x = Matrix<double>.Build.DenseOfArray(new double[,]
            {
                {3, 5},
                {5, 1},
                {10, 2}
            });
            var y = Matrix<double>.Build.DenseOfArray(new double[,] { { 0.75 }, { 0.82 }, { 0.93 } });

            var nn = new NeuralNetwork(InputLayerSize, OutputLayerSize, HiddenLayerSize);
            var cost1 = nn.ConstFunctionPrime(x, y);
            nn.Print("djW1", cost1.djW1);
            nn.Print("djW2", cost1.djW2);

            Console.Write("Press a key...");
            Console.ReadKey();
        }

        private static void TrySomething()
        {
            var x = Matrix<double>.Build.DenseOfArray(new double[,]
            {
                {1, 2},
                {3, 4}
            });
            var y = Matrix<double>.Build.DenseOfArray(new double[,]
            {
                {5, 6},
                {7, 8}
            });

            var s1 = x * y;
            Console.WriteLine(s1.ToString());
            var s2 = x.Multiply(y);
            Console.WriteLine(s2.ToString());
            Console.WriteLine(x.PointwiseMultiply(y).ToString());
            
        }
    }
}

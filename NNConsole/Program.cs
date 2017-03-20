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
            var y = Matrix<double>.Build.DenseOfArray(new double[,] { { 75 }, { 82 }, { 93 } });

            var nn = new NeuralNetwork(InputLayerSize, OutputLayerSize, HiddenLayerSize);
            var result = nn.ConstFunctionPrime(x, y);
 
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

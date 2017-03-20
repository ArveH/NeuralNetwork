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

            var x = Matrix<double>.Build.DenseOfArray(new[,]
            {
                {3.0, 5.0},
                {5.0, 1.0},
                {10.0, 2.0}
            });
            var y = Matrix<double>.Build.DenseOfArray(new[,] { { 0.75 }, { 0.82 }, { 0.93 } });

            var nn = new NeuralNetwork(InputLayerSize, OutputLayerSize, HiddenLayerSize);

            var initialScore = nn.ConstFunction(x, y);
            double scalar = 3;
            Console.Write("How many iterations: ");
            int iterations;
            if (!int.TryParse(Console.ReadLine(), out iterations)) iterations = 1000;
            Console.WriteLine($"Reducing cost in {iterations} iterations...");
            for (int i = 0; i < iterations; i++)
            {
                var costPrime = nn.ConstFunctionPrime(x, y);

                nn.W1 = nn.W1 - costPrime.dJdW1.Map(v => v * scalar);
                nn.W2 = nn.W2 - costPrime.dJdW2.Map(v => v * scalar);

                Console.Write(".");
            }
            Console.WriteLine($"\n\nInitial cost: {initialScore}");
            Console.WriteLine($"Last cost: {nn.ConstFunction(x, y)}");

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

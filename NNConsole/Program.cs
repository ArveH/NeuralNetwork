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
            var cost1 = nn.ConstFunction(x, y);
            var cost1Prime = nn.ConstFunctionPrime(x, y);
            nn.Print("djW1", cost1Prime.djW1);
            nn.Print("djW2", cost1Prime.djW2);

            double scalar = 3;
            nn.W1 = nn.W1 + cost1Prime.djW1.Map(v => v*scalar);
            nn.W2 = nn.W2 + cost1Prime.djW2.Map(v => v*scalar);

            var cost2 = nn.ConstFunction(x, y);
            Console.WriteLine($"Cost1: {cost1}");
            Console.WriteLine($"Cost2: {cost2}");

            cost1Prime = nn.ConstFunctionPrime(x, y);
            nn.W1 = nn.W1 - cost1Prime.djW1.Map(v => v * scalar);
            nn.W2 = nn.W2 - cost1Prime.djW2.Map(v => v * scalar);
            var cost3 = nn.ConstFunction(x, y);
            Console.WriteLine($"Cost2: {cost2}");
            Console.WriteLine($"Cost3: {cost3}");

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

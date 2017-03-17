using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex;

namespace NNConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            double[,] x = 
            {
                {3, 5},
                {5, 1},
                {10, 2}
            };
            double[,] y = { {75}, {82}, {93} };
            FirstNN.FirstNN.Print(Matrix<double>.Build.DenseOfArray(x), Matrix<double>.Build.DenseOfArray(y));

            Console.Write("Press a key...");
            Console.ReadKey();
        }
    }
}

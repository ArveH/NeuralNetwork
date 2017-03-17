using System;
using MathNet.Numerics.LinearAlgebra;

namespace FirstNN
{
    public class FirstNN
    {
        private int _inputLayerSize = 2;
        private int _outputLayaerSize = 1;
        private int _hiddenLayerSize = 3;

        public static void Print(Matrix<double> x, Matrix<double> y)
        {
            Console.WriteLine(x.ToString());
            Console.WriteLine(y.ToString());
        }
    }
}
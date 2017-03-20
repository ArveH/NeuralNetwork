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

        Matrix<double> _w1;
        Matrix<double> _w2;
        Matrix<double> _z2;
        Matrix<double> _a2;
        Matrix<double> _z3;
        Matrix<double> _yHat;

        public NeuralNetwork(int inputLayerSize, int outputLayaerSize, int hiddenLayerSize)
        {
            _inputLayerSize = inputLayerSize;
            _outputLayaerSize = outputLayaerSize;
            _hiddenLayerSize = hiddenLayerSize;

            _w1 = Matrix<double>.Build.Random(_inputLayerSize, _hiddenLayerSize);
            Print("w1", _w1);
            _w2 = Matrix<double>.Build.Random(_hiddenLayerSize, _outputLayaerSize);
            Print("w2", _w2);
        }

        public void Forward(Matrix<double> x)
        {
            _z2 = x * _w1;
            //Print("z2", z2);

            _a2 = Sigmoid(_z2);
            //Print("a2", a2);

            _z3 = _a2 * _w2;
            //Print("z3", z3);

            _yHat = Sigmoid(_z3);
            //Print("yHat", _yHat);
        }

        public void Print(string name, Matrix<double> m)
        {
            Console.Write($"{name}: ");
            Console.WriteLine(m.ToString());
        }

        public Matrix<double> Sigmoid(Matrix<double> m)
        {
            return m.Map(SpecialFunctions.Logistic);
        }

        public Matrix<double> SigmoidPrime(Matrix<double> m)
        {
            return m.Map(v => Math.Exp(-v) / Math.Pow(1 + Math.Exp(-v), 2));
        }

        public (Matrix<double> djW1, Matrix<double> djW2) ConstFunctionPrime(
            Matrix<double> x, 
            Matrix<double> y)
        {
            Forward(x);
            var firstPart = -(y - _yHat);
            //Print("firstPart", firstPart);
            var delta3 = firstPart.PointwiseMultiply(SigmoidPrime(_z3));
            //Print("delta3", delta3);
            var dJW2 = _a2.TransposeThisAndMultiply(delta3);
            Print("djW2", dJW2);

            var delta2 = delta3.TransposeAndMultiply(_w2).PointwiseMultiply(SigmoidPrime(_z2));
            //Print("delta2", delta2);
            var dJW1 = x.TransposeThisAndMultiply(delta2);
            Print("dJW1", dJW1);

            return (dJW1, dJW2);
        }
    }
}
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

            var rnd = new Random(0);
            _w1 = Matrix<double>
                .Build.Dense(_inputLayerSize, _hiddenLayerSize)
                .Map(v => rnd.NextDouble());
            Print("w1", _w1);
            _w2 = Matrix<double>
                .Build.Dense(_hiddenLayerSize, _outputLayaerSize)
                .Map(v => rnd.NextDouble());
            Print("w2", _w2);
        }

        public Matrix<double> Forward(Matrix<double> x)
        {
            _z2 = x * _w1;
            //Print("z2", z2);

            _a2 = Sigmoid(_z2);
            //Print("a2", a2);

            _z3 = _a2 * _w2;
            //Print("z3", z3);

            var yHat = Sigmoid(_z3);
            //Print("yHat", yHat);

            return yHat;
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

        //public Matrix<double> ConstFunction(
        //    Matrix<double> x,
        //    Matrix<double> y)
        //{
        //    var _yHat = Forward(x);

        //    var m = y - _yHat;
        //    m.Power(2).Map();
        //    var J = 0.5 * sum((y - self.yHat) * *2)
        //    return J
        //}

        public (Matrix<double> djW1, Matrix<double> djW2) ConstFunctionPrime(
            Matrix<double> x, 
            Matrix<double> y)
        {
            _yHat = Forward(x);
            var firstPart = -(y - _yHat);
            //Print("firstPart", firstPart);
            var delta3 = firstPart.PointwiseMultiply(SigmoidPrime(_z3));
            //Print("delta3", delta3);
            var dJW2 = _a2.TransposeThisAndMultiply(delta3);
            //Print("djW2", dJW2);

            var delta2 = delta3.TransposeAndMultiply(_w2).PointwiseMultiply(SigmoidPrime(_z2));
            //Print("delta2", delta2);
            var dJW1 = x.TransposeThisAndMultiply(delta2);
            //Print("dJW1", dJW1);

            return (dJW1, dJW2);
        }
    }
}
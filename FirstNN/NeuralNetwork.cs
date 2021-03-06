﻿using System;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace FirstNN
{
    public class NeuralNetwork
    {
        private readonly int _inputLayerSize;
        private readonly int _outputLayaerSize;
        private readonly int _hiddenLayerSize;

        public Matrix<double> W1 { get; set; }
        public Matrix<double> W2 { get; set; }
        Matrix<double> _z2;
        Matrix<double> _a2;
        Matrix<double> _z3;

        public NeuralNetwork(int inputLayerSize, int outputLayaerSize, int hiddenLayerSize)
        {
            _inputLayerSize = inputLayerSize;
            _outputLayaerSize = outputLayaerSize;
            _hiddenLayerSize = hiddenLayerSize;

            var rnd = new Random(0);
            W1 = Matrix<double>
                .Build.Dense(_inputLayerSize, _hiddenLayerSize)
                .Map(v => rnd.NextDouble());
            //Print("w1", W1);
            W2 = Matrix<double>
                .Build.Dense(_hiddenLayerSize, _outputLayaerSize)
                .Map(v => rnd.NextDouble());
            //Print("w2", W2);
        }

        public Matrix<double> Forward(Matrix<double> x)
        {
            _z2 = x * W1;
            //Print("z2", z2);

            _a2 = Sigmoid(_z2);
            //Print("a2", a2);

            _z3 = _a2 * W2;
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

        public double ConstFunction(
            Matrix<double> x,
            Matrix<double> y)
        {
            var yHat = Forward(x);

            var m = y - yHat;
            var m2 = m.PointwiseMultiply(m).Map(v => 0.5 * v);
            var j = m2.ColumnSums().Sum();
            return j;
        }

        public (Matrix<double> dJdW1, Matrix<double> dJdW2) ConstFunctionPrime(
            Matrix<double> x, 
            Matrix<double> y)
        {
            var yHat = Forward(x);
            var firstPart = -(y - yHat);
            //Print("firstPart", firstPart);
            var delta3 = firstPart.PointwiseMultiply(SigmoidPrime(_z3));
            //Print("delta3", delta3);
            var dJdW2 = _a2.TransposeThisAndMultiply(delta3);
            //Print("dJdW2", dJdW2);

            var delta2 = delta3.TransposeAndMultiply(W2).PointwiseMultiply(SigmoidPrime(_z2));
            //Print("delta2", delta2);
            var dJdW1 = x.TransposeThisAndMultiply(delta2);
            //Print("dJdW1", dJdW1);

            return (dJdW1, dJdW2);
        }
    }
}
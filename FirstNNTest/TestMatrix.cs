using System;
using FirstNN;
using FluentAssertions;
using MathNet.Numerics.LinearAlgebra;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace FirstNNTest
{
    [TestClass]
    public class TestMatrix
    {
        private const int InputLayerSize = 2;
        private const int OutputLayerSize = 1;
        private const int HiddenLayerSize = 3;

        private readonly Matrix<double> _x = Matrix<double>.Build.DenseOfArray(new[,]
{
                {3.0, 5.0},
                {5.0, 1.0},
                {10.0, 2.0}
            });
        private readonly Matrix<double> _y = Matrix<double>.Build.DenseOfArray(new[,] { { 0.75 }, { 0.82 }, { 0.93 } });

        private NeuralNetwork nn;

        [TestInitialize]
        public void Startup()
        {
            nn = new NeuralNetwork(InputLayerSize, OutputLayerSize, HiddenLayerSize);
        }

        [TestMethod]
        public void TestChangingAllWeights_When_AddingScalarTimesDerivative_Then_CostIncrease()
        {
            var cost1 = nn.ConstFunction(_x, _y);

            Adjusting(3.0, true);

            var cost2 = nn.ConstFunction(_x, _y);
            cost2.Should().BeGreaterThan(cost1, "because we are moving cost upwards");
        }

        [TestMethod]
        public void TestChangingAllWeights_When_SubtractingScalarTimesDerivative_Then_CostDecrease()
        {
            var cost1 = nn.ConstFunction(_x, _y);

            Adjusting(3.0);

            var cost2 = nn.ConstFunction(_x, _y);
            cost2.Should().BeLessThan(cost1, "because we are moving cost downwards");
        }

        private void Adjusting(double scalar, bool increaseCost = false)
        {
            var costPrime = nn.ConstFunctionPrime(_x, _y);
            var direction = increaseCost ? 1.0 : -1.0;
            nn.W1 = nn.W1 + direction * costPrime.dJdW1.Map(v => v * scalar);
            nn.W2 = nn.W2 + direction * costPrime.dJdW2.Map(v => v * scalar);
        }
    }
}

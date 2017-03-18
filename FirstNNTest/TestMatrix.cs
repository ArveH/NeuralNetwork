using FirstNN;
using FluentAssertions;
using MathNet.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace FirstNNTest
{
    [TestClass]
    public class TestMatrix
    {
        [TestMethod]
        public void TestArray()
        {
            NeuralNetwork.Sigmoid(0.3).Should().Be(SpecialFunctions.Logistic(0.3));
        }
    }
}

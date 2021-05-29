using CNN.AuxiliaryStructures;
using CNN.Convolution.Layers;
using Moq;
using Moq.Protected;
using System;
using Xunit;

namespace UnitTests
{
    public class UnitTest1
    {
        [Fact]
        public void Convolution_Test1()
        {
            
        }

        [Fact]
        public void Convolution_Test2()
        {

        }

        [Fact]
        public void Convolution_Test3()
        {

        }

        [Fact]
        public void Convolution_Test4()
        {

        }

        [Fact]
        public void Convolution_Test5()
        {

        }

        [Fact]
        public void Convolution_Test6()
        {

        }

        [Fact]
        public void Deconvolution_Test1()
        {

        }

        [Fact]
        public void Deconvolution_Test2()
        {

        }

        [Fact]
        public void Deconvolution_Test3()
        {

        }

        //[Fact]
        public void Deconvolution_Test11()
        {
            Tensor3D tensor = new Tensor3D(3, 3, 3);
            for (int i = 0; i < tensor.Depth; i++)
            {
                for (int j = 0; j < tensor.Height; j++)
                {
                    int v = 0;
                    for (int k = 0; k < tensor.Width; k++)
                    {
                        tensor.Set(k, j, i, v);
                    }
                }
            }

            Tensor3D[] filters = new Tensor3D[3];
            for (int f = 0; f < filters.Length; f++)
            {
                filters[f] = new Tensor3D(2, 2, 1);
                for (int i = 0; i < filters[f].Depth; i++)
                {
                    int fv = 0;
                    for (int j = 0; j < filters[f].Height; j++)
                    {
                        for (int k = 0; k < filters[f].Width; k++)
                        {
                            filters[f].Set(k, j, i, fv);
                        }
                    }
                }
            }

            float[] offsets = new float[1] { 1 };

            //DeconvolutionalLayer layer = new DeconvolutionalLayer(new Tensor3DSize(1, 2, 2), 3, 2);
            Mock<DeconvolutionalLayer> mock = new Mock<DeconvolutionalLayer>(new DeconvolutionalLayer(new Tensor3DSize(1, 2, 2), 3, 2));
            mock.Protected().SetupSet<Tensor3D[]>("Filters", filters);
            mock.Protected().SetupSet<float[]>("Offsets", offsets);

            Tensor3D result = mock.Object.Transfer(tensor);

            Assert.Equal(GetResult(0).W, result.W);
        }

        private Tensor3D GetResult(int index)
        {
            Tensor3D[] results = new Tensor3D[3];

            results[0].W = new float[]
            {
                1, 7, 10, 13, 19, 1,
                1, 1,  4,  1,  7, 1,
                1, 7, 10, 13, 19, 1,
                1, 1, 4, 1, 7, 1,
                1, 7, 10, 13, 19, 1,
                1, 1, 1, 1, 1, 1
            };

            return results[index];
        }
    }
}

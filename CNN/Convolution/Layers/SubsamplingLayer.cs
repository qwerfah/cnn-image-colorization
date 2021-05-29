using CNN.AuxiliaryStructures;
using CNN.Convolution.Layers.Functions;
using System;

namespace CNN.Convolution.Layers
{
    [Serializable]
    public class SubsamplingLayer : Layer
    {
        private readonly int _scaleX;
        private readonly int _scaleY;
        private readonly PoolingFunction _pooling;

        public SubsamplingLayer(PoolingType pooling, int stepX = 2, int stepY = 2)
        {
            if (stepX < 1 || stepY < 1)
            {
                throw new ArgumentException("Размеры окна пулинга не могут быть меньше 1!");
            }

            _scaleX = stepX;
            _scaleY = stepY;
            _pooling = PoolingFunctions.Functions[(int)pooling];
        }

        public override Tensor3D BackPropagation(Tensor3D nextLayerDeltas, Tensor3D input, Tensor3D output)
        {
            throw new NotImplementedException();
        }

        public override Tensor3D CalcOutputDeltas(Tensor3D result, Tensor3D correct)
        {
            throw new NotImplementedException("Подвыборочный слой не может быть последним!");
        }

        public override Tensor3DSize CalcSizes(Tensor3DSize inputSize)
        {
            _inputSize = inputSize;
            _outputSize = new Tensor3DSize(inputSize.Depth, inputSize.Height / _scaleY, inputSize.Width / _scaleX);

            return _outputSize;
        }

        // Подвыборка
        public override Tensor3D Transfer(Tensor3D data)
        {
            //return _pooling(data, _scaleX, _scaleY);
            throw new NotImplementedException("Подвыборочный слой не может быть последним!");
        }


    }
}

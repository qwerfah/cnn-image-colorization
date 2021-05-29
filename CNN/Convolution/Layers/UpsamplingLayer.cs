using CNN.AuxiliaryStructures;
using CNN.Convolution.Layers.Functions;
using System;

namespace CNN.Convolution.Layers
{
    [Serializable]
    class UpsamplingLayer : Layer
    {
        private readonly int _xScale;
        private readonly int _yScale;
        private readonly UpsamplingFunction _upsampling;

        public UpsamplingLayer(UpsamplingType upsampling, int xScale, int yScale)
        {
            if (xScale < 1 || yScale < 1)
            {
                throw new ArgumentException("Коэффициенты масштабирования не могут быть меньше 1!");
            }

            _upsampling = UpsamplingFunctions.Functions[(int)upsampling];
            _xScale = xScale;
            _yScale = yScale;
        }

        public override Tensor3D Transfer(Tensor3D data)
        {
            throw new NotImplementedException();
            //return _upsampling(data, _xScale, _yScale);
        }

        public override Tensor3D BackPropagation(Tensor3D nextLayerDeltas, Tensor3D input, Tensor3D output)
        {
            Tensor3D deltas = new Tensor3D(_inputSize.Width, _inputSize.Height, _inputSize.Depth);
            float count = _xScale * _yScale;

            for (int d = 0; d < _inputSize.Depth; d++)
            {
                for (int y = 0; y < _inputSize.Height; y++)
                {
                    for (int x = 0; x < _inputSize.Width; x++)
                    {
                        float delta = 0;
                        int i0 = y * _yScale;
                        int j0 = x * _xScale;

                        for (int i = i0; i < i0 + _yScale; i++)
                        {
                            for (int j = j0; j < j0 + _xScale; j++)
                            {
                                delta += nextLayerDeltas.Get(j, i, d);
                            }
                        }

                        deltas.Set(x, y, d, delta / count);
                    }
                }
            }

            return deltas;
        }

        public override Tensor3DSize CalcSizes(Tensor3DSize inputSize)
        {
            int height = inputSize.Height * _yScale;
            int width = inputSize.Width * _xScale;

            _inputSize = inputSize;
            _outputSize = new Tensor3DSize(inputSize.Depth, height, width);

            return _outputSize;
        }

        public override Tensor3D CalcOutputDeltas(Tensor3D result, Tensor3D correct)
        {
            throw new System.NotImplementedException();
        }
    }
}

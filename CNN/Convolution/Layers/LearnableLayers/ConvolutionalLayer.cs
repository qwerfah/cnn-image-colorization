using CNN.AuxiliaryStructures;
using System;
using System.Numerics;
using System.Threading.Tasks;

namespace CNN.Convolution.Layers
{
    /// <summary>
    /// Слой свертки.
    /// </summary>
    [Serializable]
    public class ConvolutionalLayer : LearnableLayer
    {
        private int _dilation = 1;

        public ConvolutionalLayer(Tensor3DSize filterSize, int filtersCount = 1,
                                  int strides = 1, int dilation = 1)
           : base(filterSize, filtersCount, filtersCount, strides)
        {
            _dilation = dilation;
        }

        public override Tensor3D Transfer(Tensor3D data)
        {
            int outputDepth = Filters.Length;
            Tensor3D maps = new Tensor3D(_outputSize.Width, _outputSize.Height, outputDepth);
            int dilation_1 = _dilation - 1;
            int outputWidth = _outputSize.Width;
            int outputHeight = _outputSize.Height;
            int dataWidth = data.Width;
            int dataHeight = data.Height;
            int dataDepth = data.Depth;

            Parallel.For(0, outputDepth, d =>
            {
                Tensor3D filter = Filters[d];
                float offset = Offsets[d];

                for (int ay = 0; ay < outputHeight; ay++)
                {
                    int y = ay * _strides - (filter.Height * _dilation + dilation_1) / 2;

                    for (int ax = 0; ax < outputWidth; ax++)
                    {
                        int x = ax * _strides - (filter.Width * _dilation + dilation_1) / 2;
                        float a = offset;

                        for (int fy = 0; fy < filter.Height; fy++)
                        {
                            int oy = y + fy * _dilation + dilation_1;

                            for (int fx = 0; fx < filter.Width; fx++)
                            {
                                int ox = x + fx * _dilation + dilation_1;

                                if ((oy >= 0) && (oy < dataHeight) && (ox >= 0) && (ox < dataWidth))
                                {
                                    int fi = ((filter.Width * fy) + fx) * filter.Depth;
                                    int ti = ((dataWidth * oy) + ox) * dataDepth;

                                    a += DotVectors(data.W, filter.W, ti, fi, filter.Depth);
                                }
                            }
                        }
                        maps.Set(ax, ay, d, a);
                    }
                }
            });

            return maps;
        }

        public override Tensor3D CalcOutputDeltas(Tensor3D result, Tensor3D correct)
        {
            Tensor3D deltas = new Tensor3D(_outputSize.Width, _outputSize.Height, _outputSize.Depth);
            float delta = 0.0f;

            Parallel.For(0, _outputSize.Depth, d =>
            {
                for (int y = 0; y < _outputSize.Height; y++)
                {
                    for (int x = 0; x < _outputSize.Width; x++)
                    {
                        float dL = CrossEntropyError(correct.Get(x, y, d), result.Get(x, y, d));
                        delta += dL;
                        deltas.Set(x, y, d, dL);
                    }
                }
            });

            float count = _outputSize.Width * _outputSize.Height * _outputSize.Depth;
            Console.WriteLine($"Loss: {delta / count}");

            return deltas;
        }

        /// <summary>
        /// Средняя квадратичная функция ошибки.
        /// </summary>
        /// <param name="t">Истинное значение.</param>
        /// <param name="y">Полученное значение.</param>
        /// <returns></returns>
        public static float AverageSquareError(float t, float y)
        {
            return (y - t);
        }

        /// <summary>
        /// Категориальная перекрестная энтропия.
        /// </summary>
        /// <param name="t">Истинное значение.</param>
        /// <param name="y">Полученное значение.</param>
        /// <returns></returns>
        public static float CrossEntropyError(float t, float y)
        {
            return -t / y + (1.0f - t) / (1.0f - y);
        }

        public override Tensor3D BackPropagation(Tensor3D nextLayerDeltas, Tensor3D input, Tensor3D output)
        {
            int outputDepth = Filters.Length;
            int dilation_1 = _dilation - 1;

            int outputWidth = _outputSize.Width;
            int outputHeight = _outputSize.Height;

            int inputWidth = _inputSize.Width;
            int inputHeight = _inputSize.Height;
            int inputDepth = _inputSize.Depth;

            Parallel.For(0, outputDepth, d =>
            {
                Tensor3D deltaW = _deltaWeights[d];

                for (int ay = 0; ay < outputHeight; ay++)
                {
                    int y = ay * _strides - (deltaW.Height * _dilation + dilation_1) / 2;

                    for (int ax = 0; ax < outputWidth; ax++)
                    {
                        int x = ax * _strides - (deltaW.Width * _dilation + dilation_1) / 2;
                        float delta = nextLayerDeltas.Get(ax, ay, d);
                        _deltaOffsets[d] += delta;

                        for (int fy = 0; fy < deltaW.Height; fy++)
                        {
                            int oy = y + fy * _dilation + dilation_1;

                            for (int fx = 0; fx < deltaW.Width; fx++)
                            {
                                int ox = x + fx * _dilation + dilation_1;

                                if ((oy >= 0) && (oy < inputHeight) && (ox >= 0) && (ox < inputWidth))
                                {
                                    int fi = ((deltaW.Width * fy) + fx) * deltaW.Depth;
                                    int ti = ((inputWidth * oy) + ox) * inputDepth;

                                    Mult(input.W, deltaW.W, ti, fi, deltaW.Depth, delta);

                                    /*
                                    for (int fd = 0; fd < deltaW.Depth; fd++)
                                    {
                                        deltaW.W[fi + fd] += input.W[ti + fd] * delta;
                                    }
                                    */
                                }
                            }
                        }
                    }
                }
            });

            Tensor3D deltas = new Tensor3D(inputWidth, inputHeight, inputDepth);
            int fw_2 = -_filterWidth / 2;
            int fh_2 = -_filterHeight / 2;

            Parallel.For(0, nextLayerDeltas.Depth, d =>
            {
                Tensor3D filter = Filters[d];
                int x = fw_2;
                int y = fh_2;

                for (int ay = 0; ay < nextLayerDeltas.Height; y += _strides, ay++)
                {
                    x = fw_2;
                    for (int ax = 0; ax < nextLayerDeltas.Width; x += _strides, ax++)
                    {
                        float value = nextLayerDeltas.Get(ax, ay, d);
                        for (int fy = 0; fy < filter.Height; fy++)
                        {
                            int oy = y + fy;
                            for (int fx = 0; fx < filter.Width; fx++)
                            {
                                int ox = x + fx;
                                if ((oy >= 0) && (oy < inputHeight) && (ox >= 0) && (ox < inputWidth))
                                {
                                    int ix1 = ((inputWidth * oy) + ox) * inputDepth;
                                    int ix2 = ((filter.Width * fy) + fx) * filter.Depth;

                                    Mult(filter.W, deltas.W, ix2, ix1, filter.Depth, value);

                                    /*
                                    for (int fd = 0; fd < filter.Depth; fd++)
                                    {
                                        
                                        deltas.W[ix1] += (filter.W[ix2] * value);
                                    }
                                    */
                                }
                            }
                        }
                    }
                }
            });

            return deltas;
        }

        public override Tensor3DSize CalcSizes(Tensor3DSize inputSize)
        {
            if (inputSize.Depth != _filterDepth)
            {
                throw new ArgumentException("Глубина входа не совпадает с глубиной фильтра!");
            }

            int height = inputSize.Height / _strides;
            int width = inputSize.Width / _strides;

            _inputSize = inputSize;
            _outputSize = new Tensor3DSize(_filtersCount, height, width);

            return _outputSize;
        }
    }
}

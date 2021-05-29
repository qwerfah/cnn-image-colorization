using CNN.AuxiliaryStructures;
using System;
using System.Numerics;
using System.Threading.Tasks;

namespace CNN.Convolution.Layers
{
    /// <summary>
    /// Слой обратной свертки.
    /// </summary>
    [Serializable]
    public class DeconvolutionalLayer : LearnableLayer
    {
        public DeconvolutionalLayer(Tensor3DSize filterSize, int filtersCount = 1, int strides = 2)
            : base(filterSize, filtersCount, filterSize.Depth, strides)
        { }

        public override Tensor3D Transfer(Tensor3D data)
        {
            int outputWidth = _outputSize.Width;
            int outputHeight = _outputSize.Height;
            int outputDepth = _outputSize.Depth;
            Tensor3D maps = new Tensor3D(outputWidth, outputHeight, outputDepth);

            int fw_2 = -_filterWidth / 2;
            int fh_2 = -_filterHeight / 2;

            Parallel.For(0, data.Depth, d =>
            {
                Tensor3D filter = Filters[d];
                int x = fw_2;
                int y = fh_2;

                for (int ay = 0; ay < data.Height; y += _strides, ay++)
                {
                    x = fw_2;
                    for (int ax = 0; ax < data.Width; x += _strides, ax++)
                    {
                        float value = data.Get(ax, ay, d);
                        for (int fy = 0; fy < filter.Height; fy++)
                        {
                            int oy = y + fy;
                            for (int fx = 0; fx < filter.Width; fx++)
                            {
                                int ox = x + fx;
                                if ((oy >= 0) && (oy < outputHeight) && (ox >= 0) && (ox < outputWidth))
                                {
                                    int ix1 = ((outputWidth * oy) + ox) * outputDepth;
                                    int ix2 = ((filter.Width * fy) + fx) * filter.Depth;

                                    Mult(filter.W, maps.W, ix2, ix1, filter.Depth, value);
                                }
                            }
                        }
                    }
                }
            });

            Parallel.For(0, outputDepth, d =>
            {
                float offset = Offsets[d];

                for (int y = 0; y < outputHeight; y++)
                {
                    for (int x = 0; x < outputWidth; x++)
                    {
                        maps.Add(x, y, d, offset);
                    }
                }
            });

            return maps;
        }

        public override Tensor3D BackPropagation(Tensor3D nextLayerDeltas, Tensor3D input, Tensor3D output)
        {
            int outputWidth = _outputSize.Width;
            int outputHeight = _outputSize.Height;
            int outputDepth = _outputSize.Depth;

            Parallel.For(0, input.Depth, d =>
            {
                Tensor3D deltaW = _deltaWeights[d];
                int x = -deltaW.Width / 2;
                int y = -deltaW.Height / 2;

                for (int ay = 0; ay < input.Height; y += _strides, ay++)
                {
                    x = -deltaW.Width / 2;
                    for (int ax = 0; ax < input.Width; x += _strides, ax++)
                    {
                        float value = input.Get(ax, ay, d);
                        for (int fy = 0; fy < deltaW.Height; fy++)
                        {
                            int oy = y + fy;
                            for (int fx = 0; fx < deltaW.Width; fx++)
                            {
                                int ox = x + fx;
                                if ((oy >= 0) && (oy < outputHeight) && (ox >= 0) && (ox < outputWidth))
                                {
                                    int ix1 = ((outputWidth * oy) + ox) * outputDepth;
                                    int ix2 = ((deltaW.Width * fy) + fx) * deltaW.Depth;

                                    Mult(nextLayerDeltas.W, deltaW.W, ix1, ix2, deltaW.Depth, value);

                                    /*
                                    for (int fd = 0; fd < deltaW.Depth; fd++)
                                    {
                                        deltaW.W[ix2 + fd] += nextLayerDeltas.W[ix1 + fd] * value;
                                    }
                                    */
                                }
                            }
                        }
                    }
                }
            });

            Parallel.For(0, outputDepth, d =>
            {
                for (int y = 0; y < outputHeight; y++)
                {
                    for (int x = 0; x < outputWidth; x++)
                    {
                        _deltaOffsets[d] += nextLayerDeltas.Get(x, y, d);
                    }
                }
            });

            int depth = Filters.Length;
            Tensor3D deltas = new Tensor3D(_inputSize.Width, _inputSize.Height, depth);

            Parallel.For(0, depth, (int d) =>
            {
                Tensor3D filter = Filters[d];

                for (int ay = 0; ay < deltas.Height; ay++)
                {
                    int y = ay * _strides - filter.Height / 2;

                    for (int ax = 0; ax < deltas.Width; ax++)
                    {
                        int x = ax * _strides - filter.Width / 2;
                        float a = 0.0f;

                        for (int fy = 0; fy < filter.Height; fy++)
                        {
                            int oy = y + fy + 1;

                            for (int fx = 0; fx < filter.Width; fx++)
                            {
                                int ox = x + fx + 1;

                                if ((oy >= 0) && (oy < nextLayerDeltas.Height) && (ox >= 0) && (ox < nextLayerDeltas.Width))
                                {
                                    int fi = ((filter.Width * fy) + fx) * filter.Depth;
                                    int ti = ((nextLayerDeltas.Width * oy) + ox) * nextLayerDeltas.Depth;

                                    a += DotVectors(nextLayerDeltas.W, filter.W, ti, fi, filter.Depth);

                                    /*
                                    for (int fd = 0; fd < filter.Depth; fd++)
                                    {
                                        a += filter.W[fi + fd] * nextLayerDeltas.W[ti + fd];
                                    }
                                    */
                                }
                            }
                        }
                        deltas.Set(ax, ay, d, a);
                    }
                }
            });

            return deltas;
        }

        public override Tensor3D CalcOutputDeltas(Tensor3D result, Tensor3D correct)
        {
            throw new NotImplementedException();
        }

        public override Tensor3DSize CalcSizes(Tensor3DSize inputSize)
        {
            if (inputSize.Depth != _filtersCount)
            {
                throw new ArgumentException("Глубина входа не совпадает с глубиной фильтра!");
            }

            int height = _strides * inputSize.Height;
            int width = _strides * inputSize.Width;
            _inputSize = inputSize;

            _outputSize = new Tensor3DSize(_filterDepth, height, width);

            return _outputSize;
        }
    }
}

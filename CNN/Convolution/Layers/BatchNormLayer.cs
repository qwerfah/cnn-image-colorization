using CNN.AuxiliaryStructures;
using MathNet.Numerics.Distributions;
using System;
using System.IO;
using System.Threading.Tasks;

namespace CNN.Convolution.Layers
{
    /// <summary>
    /// Слой пакетной нормализации.
    /// </summary>
    [Serializable]
    public class BatchNormLayer : Layer
    {
        private float[] _mean = null;
        private float[] _variance = null;
        private float _scale = 1.0f;
        private float _deltaScale = 0.0f;
        // private float _offset = 0.0f;
        private float _epsilon = 1e-5f;

        public BatchNormLayer()
        {
            _variance = new float[512];
            _mean = new float[512];
            Normal normal = new Normal();
            _scale = (float)normal.Sample();
            // _offset = (float)normal.Sample();
        }

        /// <summary>
        /// Загрузка весов из файла.
        /// </summary>
        /// <param name="reader">Объект для считывания бинарных данных в потоке.</param>
        public void LoadWeights(BinaryReader reader)
        {
            _mean = new float[_inputSize.Depth];
            _variance = new float[_inputSize.Depth];

            for (int i = 0; i < _inputSize.Depth; i++)
            {
                _mean[i] = reader.ReadSingle();
            }

            for (int i = 0; i < _inputSize.Depth; i++)
            {
                _variance[i] = reader.ReadSingle();
            }

            _scale = reader.ReadSingle();
        }

        public override Tensor3D Transfer(Tensor3D data)
        {
            Tensor3D normalized =
                new Tensor3D(_outputSize.Width, _outputSize.Height, _outputSize.Depth);
            int height = _outputSize.Height;
            int width = _outputSize.Width;

            /*
            Parallel.For(0, _outputSize.Depth, d =>
            {
                float mean = 0.0, variance = 0.0;
                for (int y = 0; y < _outputSize.Height; y++)
                {
                    for (int x = 0; x < _outputSize.Width; x++)
                    {
                        mean += data.Get(x, y, d);
                    }
                }
                mean = mean * _scale / count;
                for (int y = 0; y < _outputSize.Height; y++)
                {
                    for (int x = 0; x < _outputSize.Width; x++)
                    {
                        variance += Math.Pow(data.Get(x, y, d) - mean, 2);
                    }
                }
                variance = Math.Sqrt(variance / count + _epsilon);
                for (int y = 0; y < _outputSize.Height; y++)
                {
                    for (int x = 0; x < _outputSize.Width; x++)
                    {
                        normalized.Set(x, y, d, ((data.Get(x, y, d) * _scale - mean) / variance) + _offset);
                    }
                }
            });
            */

            Parallel.For(0, data.Depth, d =>
            {
                double div = Math.Sqrt(_epsilon + _variance[d] / _scale);
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        normalized.Set(x, y, d, (float)((data.Get(x, y, d) - _mean[d] / _scale) / div));
                    }
                }
            });

            return normalized;
        }

        public override Tensor3D BackPropagation(Tensor3D nextLayerDeltas, Tensor3D input, Tensor3D output)
        {
            Tensor3D deltas = new Tensor3D(_outputSize.Width, _outputSize.Height, _outputSize.Depth);
            float m = _outputSize.Width * _outputSize.Height;

            // Вычисление градиента по _scale
            Parallel.For(0, nextLayerDeltas.Depth, d =>
            {
                float mean = _mean[d], variance = _variance[d];
                float sum = _epsilon + variance / _scale;
                float pow = (float)Math.Pow(sum, 3.0 / 2.0);
                float derivative = (2.0f * mean * sum + variance) / 
                    (2.0f * _scale * _scale * pow);
                float div = -1.0f / (2.0f * pow);
                float m_s = mean / _scale;

                float dL_dVar = 0.0f;
                float dVar_dMean = 0.0f;

                for (int y = 0; y < nextLayerDeltas.Height; y++)
                {
                    for (int x = 0; x < nextLayerDeltas.Width; x++)
                    {
                        float dl_Yi = nextLayerDeltas.Get(x, y, d);
                        float xi = input.Get(x, y, d);

                        _deltaScale += (dl_Yi * derivative);
                        dL_dVar += (dl_Yi * (xi - m_s) * div);
                        dVar_dMean -= (xi - mean);
                    }
                }

                dVar_dMean *= (2.0f / m);

                float dL_dMean = 0.0f;
                float sqrt = (float)Math.Sqrt(sum);
                float dY_dMean = -1.0f / (_scale * sqrt);

                for (int y = 0; y < nextLayerDeltas.Height; y++)
                {
                    for (int x = 0; x < nextLayerDeltas.Width; x++)
                    {
                        dL_dMean += (nextLayerDeltas.Get(x, y, d) * dY_dMean + dL_dVar * dVar_dMean);
                    }
                }

                for (int y = 0; y < nextLayerDeltas.Height; y++)
                {
                    for (int x = 0; x < nextLayerDeltas.Width; x++)
                    {
                        deltas.Set(x, y, d, nextLayerDeltas.Get(x, y, d) / sqrt + dL_dMean / m + dL_dVar * 2.0f / m * (input.Get(x, y, d) - mean));
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
            _inputSize = inputSize;
            _outputSize = inputSize;

            return inputSize;
        }

        /// <summary>
        /// Обновление обучаемых параметров.
        /// </summary>
        /// <param name="rate">Скорость обучения (коэффициент шага градиента).</param>
        public void UpdateWeights(float rate)
        {
            _scale -= _deltaScale * rate;
            _deltaScale = 0.0f;
        }
    }
}

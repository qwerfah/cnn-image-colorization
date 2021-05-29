using CNN.AuxiliaryStructures;
using MathNet.Numerics.Distributions;
using System;
using System.IO;
using System.Numerics;
using System.Threading.Tasks;

namespace CNN.Convolution.Layers
{
    /// <summary> Обучаемый слой сверточной нейронной сети. </summary>
    [Serializable]
    public abstract class LearnableLayer : Layer
    {
        /// <summary> Веса фильтров. </summary>
        public Tensor3D[] Filters { get; protected set; }
        /// <summary> Веса смещения. </summary>
        public float[] Offsets { get; protected set; }

        /// <summary> Градиенты весов по входам. </summary>
        protected Tensor3D[] _deltaWeights = null;
        /// <summary> Градиенты смещений по входам. </summary>
        protected float[] _deltaOffsets = null;

        /// <summary> 
        /// Предыдущие значения градиентов весов по входам
        /// для использования в алгоритме оптимизации Adam.
        /// </summary>
        protected Tensor3D[] _prevDeltaWeights = null;
        /// <summary> 
        /// Предыдущие значения градиентов смещений по входам
        /// для использования в алгоритме оптимизации Adam.
        /// </summary>
        protected float[] _prevDeltaOffsets = null;

        /// <summary> 
        /// Значения скользящего среднего второго момента градиента весов
        /// для использования в алгоритме оптимизации Adam.
        /// </summary>
        protected Tensor3D[] _momentumWeights = null;
        /// <summary> 
        /// Значения скользящего среднего второго момента градиента смещений
        /// для использования в алгоритме оптимизации Adam.
        /// </summary>
        protected float[] _momentumOffsets = null;

        protected float _epsilon = 1e-8f;

        protected int _offsetsCount = 0;
        protected int _filtersCount = 0;

        protected int _filterDepth = 0;
        protected int _filterHeight = 0;
        protected int _filterWidth = 0;

        protected int _strides = 1;

        public LearnableLayer(Tensor3DSize filterSize, int filtersCount, int offsetsCount, int strides)
        {
            if (filterSize.Height < 1 || filterSize.Width < 1 || filterSize.Depth < 1 || filtersCount < 1 || strides < 1)
            {
                throw new ArgumentException("Размеры ядра, глубина входа или число страйдов и фильтров не могут быть меньше 1!");
            }

            _offsetsCount = offsetsCount;
            _filtersCount = filtersCount;

            _filterDepth = filterSize.Depth;
            _filterHeight = filterSize.Height;
            _filterWidth = filterSize.Width;

            AlocateParameters();
            GenerateRandomWeights();

            _strides = strides;
        }

        /// <summary> Выделение памяти для параметров и градиентов. </summary>
        public void AlocateParameters()
        {
            Filters = new Tensor3D[_filtersCount];
            Offsets = new float[_offsetsCount];

            _deltaWeights = new Tensor3D[_filtersCount];
            _deltaOffsets = new float[_offsetsCount];

            _prevDeltaWeights = new Tensor3D[_filtersCount];
            _prevDeltaOffsets = new float[_offsetsCount];

            _momentumWeights = new Tensor3D[_filtersCount];
            _momentumOffsets = new float[_offsetsCount];

            for (int f = 0; f < _filtersCount; f++)
            {
                Filters[f] = new Tensor3D(_filterWidth, _filterHeight, _filterDepth);
                _deltaWeights[f] = new Tensor3D(_filterWidth, _filterHeight, _filterDepth);
                _prevDeltaWeights[f] = new Tensor3D(_filterWidth, _filterHeight, _filterDepth);
                _momentumWeights[f] = new Tensor3D(_filterWidth, _filterHeight, _filterDepth);
            }
        }

        /// <summary> Загрузка весов фильтров и смещений. </summary>
        public void LoadWeights(BinaryReader reader)
        {
            for (int f = 0; f < _filtersCount; f++)
            {
                for (int d = 0; d < _filterDepth; d++)
                {
                    for (int y = 0; y < _filterHeight; y++)
                    {
                        for (int x = 0; x < _filterWidth; x++)
                        {
                            Filters[f].Set(x, y, d, reader.ReadSingle());
                        }
                    }
                }
            }

            for (int f = 0; f < _offsetsCount; f++)
            {
                Offsets[f] = reader.ReadSingle();
            }
        }

        /// <summary> Генерация весов фильтров и смещений. </summary>
        protected void GenerateRandomWeights()
        {
            Normal normalDist = new Normal();

            float t = (float)Math.Sqrt(2.0 / (float)(_filterHeight * _filterWidth * _filterDepth));

            for (int f = 0; f < _filtersCount; f++)
            {
                for (int d = 0; d < _filterDepth; d++)
                {
                    for (int y = 0; y < _filterHeight; y++)
                    {
                        for (int x = 0; x < _filterWidth; x++)
                        {
                            Filters[f].Set(x, y, d, t * (float)normalDist.Sample());
                        }
                    }
                }
            }

            for (int f = 0; f < _offsetsCount; f++)
            {
                Offsets[f] = t * (float)normalDist.Sample();
            }
        }

        /// <summary> 
        /// Обновление значений весов фильтров и смещений 
        /// по алгоритму стахостического градиентного спуска. 
        /// </summary>
        public void UpdateWeightsSGD(float rate)
        {
            Parallel.For(0, _filtersCount, f =>
            {
                for (int y = 0; y < _filterHeight; y++)
                {
                    for (int x = 0; x < _filterWidth; x++)
                    {
                        for (int d = 0; d < _filterDepth; d++)
                        {
                            if (float.IsNaN(_deltaWeights[f].Get(x, y, d)) || float.IsInfinity(_deltaWeights[f].Get(x, y, d)))
                            {
                                throw new ArithmeticException("Ошибка вычисления изменения весов!");
                            }
                            Filters[f].Add(x, y, d, -rate * _deltaWeights[f].Get(x, y, d));
                            _deltaWeights[f].Set(x, y, d, 0.0f);
                        }
                    }
                }
            });

            for (int d = 0; d < _offsetsCount; d++)
            {
                Offsets[d] -= (rate * _deltaOffsets[d]);
                _deltaOffsets[d] = 0.0f;
            }
        }

        /// <summary> 
        /// Обновление значений весов фильтров и смещений 
        /// по алгоритму градиентного спуска с импульсом. 
        /// </summary>
        public void UpdateWeightsPulse(float rate, float pulse)
        {
            Parallel.For(0, _filtersCount, f =>
            {
                for (int y = 0; y < _filterHeight; y++)
                {
                    for (int x = 0; x < _filterWidth; x++)
                    {
                        for (int d = 0; d < _filterDepth; d++)
                        {
                            if (float.IsNaN(_deltaWeights[f].Get(x, y, d)) || float.IsInfinity(_deltaWeights[f].Get(x, y, d)))
                            {
                                throw new ArithmeticException("Ошибка вычисления изменения весов!");
                            }

                            float dW = _prevDeltaWeights[f].Get(x, y, d) * pulse - rate * _deltaWeights[f].Get(x, y, d);
                            Filters[f].Add(x, y, d, dW);
                            _prevDeltaWeights[f].Set(x, y, d, dW);
                            _deltaWeights[f].Set(x, y, d, 0.0f);
                        }
                    }
                }
            });

            for (int d = 0; d < _offsetsCount; d++)
            {
                Offsets[d] -= (rate * _deltaOffsets[d]);
                _deltaOffsets[d] = 0.0f;
            }
        }

        /// <summary> 
        /// Обновление значений весов фильтров и смещений 
        /// по алгоритму градиентного спуска с импульсом и моментом (опитмизатор Adam). 
        /// </summary>
        public void UpdateWeightsAdam(float rate, float pulse, float momentum)
        {
            Parallel.For(0, _filtersCount, f =>
            {
                for (int y = 0; y < _filterHeight; y++)
                {
                    for (int x = 0; x < _filterWidth; x++)
                    {
                        for (int d = 0; d < _filterDepth; d++)
                        {
                            if (float.IsNaN(_deltaWeights[f].Get(x, y, d)) || float.IsInfinity(_deltaWeights[f].Get(x, y, d)))
                            {
                                throw new ArithmeticException("Ошибка вычисления изменения весов!");
                            }

                            float dW = _deltaWeights[f].Get(x, y, d);
                            float mi = pulse * _prevDeltaWeights[f].Get(x, y, d) + (1.0f - pulse) * dW;
                            float vi = momentum * _momentumWeights[f].Get(x, y, d) + (1.0f - momentum) * dW * dW;

                            _prevDeltaWeights[f].Set(x, y, d, mi);
                            _momentumWeights[f].Set(x, y, d, vi);

                            mi /= (1.0f - pulse);
                            vi /= (1.0f - momentum);

                            Filters[f].Add(x, y, d, -rate * mi / (float)Math.Sqrt(vi + _epsilon));
                            _deltaWeights[f].Set(x, y, d, 0.0f);
                        }
                    }
                }
            });

            for (int d = 0; d < _offsetsCount; d++)
            {
                float dW = _deltaOffsets[d];
                float mi = pulse * _prevDeltaOffsets[d] + (1.0f - pulse) * dW;
                float vi = momentum * _momentumOffsets[d] + (1.0f - momentum) * dW * dW;

                _prevDeltaOffsets[d] = mi;
                _momentumOffsets[d] = vi;

                mi /= (1.0f - pulse);
                vi /= (1.0f - momentum);

                Offsets[d] -= (rate * mi / (float)Math.Sqrt(vi + _epsilon));
                _deltaOffsets[d] = 0.0f;
            }
        }

        /// <summary>
        /// Умножение массива на число и прибавление результата к элементам другого массива (SIMD).
        /// </summary>
        /// <param name="input">Массив входных данных.</param>
        /// <param name="output">Массив для сохранения результата.</param>
        /// <param name="fromInput">Индекс начало участка входного массива.</param>
        /// <param name="fromOutput">Индекс начало участка выходного массива.</param>
        /// <param name="count">Длинна участка массивов.</param>
        /// <param name="value">Значение, на которое умножается входной массив.</param>
        protected static void Mult(float[] input, float[] output, int fromInput, int fromOutput, int count, float value)
        {
            int vectorSize = Vector<float>.Count;
            Vector<float> mask = new Vector<float>(value);
            int i, j;
            int to = fromInput + count;

            for (i = fromInput, j = fromOutput; i < to - vectorSize; i += vectorSize, j += vectorSize)
            {
                Vector<float> vector = new Vector<float>(input, i);
                Vector<float> result = Vector.Multiply(vector, mask);
                for (int k = j, l = 0; k < j + vectorSize; k++, l++)
                {
                    output[k] += result[l];
                }
            }

            for (; i < to; i++, j++)
            {
                output[j] += input[i] * value;
            }
        }

        /// <summary>
        /// Скалярное произведение двух векторов (SIMD).
        /// </summary>
        /// <param name="array">Массив данных.</param>
        /// <param name="weights">Массив весов.</param>
        /// <param name="fromArray">Индекс начала участка массива данных.</param>
        /// <param name="fromWeights">Индекс начала участка массива весов.</param>
        /// <param name="count">Число элементов в скалярном произведении.</param>
        /// <returns></returns>
        protected static float DotVectors(float[] array, float[] weights, int fromArray, int fromWeights, int count)
        {
            int vectorSize = Vector<float>.Count;
            float result = 0.0f;
            int i, j;
            int to = fromArray + count;

            for (i = fromArray, j = fromWeights; i < to - vectorSize; i += vectorSize, j += vectorSize)
            {
                var v1 = new Vector<float>(array, i);
                var v2 = new Vector<float>(weights, j);
                result += Vector.Dot(v1, v2);
            }

            for (; i < to; i++, j++)
            {
                result += (array[i] * weights[j]);
            }

            return result;
        }
    }
}

using CNN.AuxiliaryStructures;
using System;
using System.Diagnostics;
using System.Numerics;
using System.Threading.Tasks;

namespace CNN.Convolution.Layers
{
    /// <summary>
    /// Слой функции активации Softmax.
    /// </summary>
    [Serializable]
    public class SoftmaxLayer : Layer
    {
        public override Tensor3D Transfer(Tensor3D data)
        {
            int width = data.Width;
            int depth = data.Depth;
            Tensor3D result = new Tensor3D(data.Width, data.Height, data.Depth);

            Parallel.For(0, data.Height, y =>
            {
                //float[] array = new float[depth];

                for (int x = 0; x < width; x++)
                {
                    int from = ((data.Width * y) + x) * data.Depth;
                    float max = Max(data.W, from, from + depth);

                    double sum = 0.0;
                    for (int d = 0; d < depth; d++)
                    {
                        float value = data.Get(x, y, d);
                        sum += Math.Exp(value - max);
                    }

                    //int from = ((data.Width * y) + x) * data.Depth;
                    //float sum = Sum(array);

                    for (int d = 0; d < depth; d++)
                    {
                        float value = data.Get(x, y, d);
                        float res = (float)(Math.Exp(value - max) / sum);
                        result.Set(x, y, d, res);
                    }
                }
            });

            return result;
        }

        /// <summary>
        /// Сумма элементов массива (SIMD).
        /// </summary>
        /// <param name="array">Входной массив.</param>
        /// <returns></returns>
        private float Sum(float[] array)
        {
            int vectorSize = Vector<float>.Count;
            var accVector = Vector<float>.Zero;
            int i;

            for (i = 0; i < array.Length - vectorSize; i += vectorSize)
            {
                var v = new Vector<float>(array, i);
                accVector = Vector.Add(accVector, v);
            }

            float result = Vector.Dot(accVector, Vector<float>.One);
            for (; i < array.Length; i++)
            {
                result += array[i];
            }

            return result;
        }

        /// <summary>
        /// Максимальное значение элементов массива (SIMD).
        /// </summary>
        /// <param name="array">Входной массив.</param>
        /// <param name="from">Начальный индекс.</param>
        /// <param name="to">Индекс конца участка массива.</param>
        /// <returns></returns>
        private float Max(float[] array, int from, int to)
        {
            int vectorSize = Vector<float>.Count;
            var prev = new Vector<float>(array, from);
            int i;

            for (i = from + vectorSize; i < to - vectorSize; i += vectorSize)
            {
                var v = new Vector<float>(array, i);
                prev = Vector.Max(prev, v);
            }

            float max = prev[0];

            for (int j = 1; j < vectorSize; j++)
            {
                if (prev[j] > max) max = prev[j];
            }
            for (; i < to; i++)
            {
                if (array[i] > max) max = array[i];
            }

            return max;
        }

        public override Tensor3D BackPropagation(Tensor3D nextLayerDeltas, Tensor3D input, Tensor3D output)
        {
            int width = nextLayerDeltas.Width;
            int depth = nextLayerDeltas.Depth;
            var deltas = new Tensor3D(nextLayerDeltas.Width, nextLayerDeltas.Height, nextLayerDeltas.Depth);

            Parallel.For(0, nextLayerDeltas.Height, y =>
            {
                for (int x = 0; x < width; x++)
                {
                    for (int d = 0; d < depth; d++)
                    {
                        float value = output.Get(x, y, d);
                        deltas.Set(x, y, d, nextLayerDeltas.Get(x, y, d) * value * (1f - value));
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
            _outputSize = _inputSize;

            return _outputSize;
        }
    }
}

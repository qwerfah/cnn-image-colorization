using CNN.AuxiliaryStructures;
using System;
using System.Runtime.CompilerServices;

namespace CNN.Convolution.Layers.Functions
{
    public enum PaddingType
    {
        Valid,
        Same,
        Full,
        None
    }

    public delegate Pair<float[,], float[,]>
        ConvolutionFunction(float[][,] matrixes, float[,,] filter, int strides, ActivationFunction activation);

    public static class ConvolutionFunctions
    {
        public static ConvolutionFunction[] Functions =
        {
            ValidPaddingConvolution,
            SamePaddingConvolution,
            FullPaddingConvolution,
        };

        // Свертка с same padding (выход - карта меньшего размера)
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Pair<float[,], float[,]>
            ValidPaddingConvolution(float[][,] matrixes, float[,,] filter, int strides, ActivationFunction activation)
        {
            int H = matrixes[0].GetLength(0), h = filter.GetLength(1);
            int W = matrixes[0].GetLength(1), w = filter.GetLength(2);
            int mh = (int)Math.Ceiling((float)(H - h) / (float)strides) + 1;
            int mw = (int)Math.Ceiling((float)(W - w) / (float)strides) + 1;
            float[,] map = new float[mh, mw];
            float[,] unactivatedMap = new float[mh, mw];

            // Цикл по глубине входа
            for (int m = 0; m < matrixes.Length; m++)
            {
                // Цикл по высоте карты признаков
                for (int i = 0, mi = 0; mi < mh; i += strides, mi++)
                {
                    // Цикл по ширине карты признаков
                    for (int j = 0, mj = 0; mj < mw; j += strides, mj++)
                    {
                        int th = (i + h <= H) ? (i + h) : H;
                        int tw = (j + w <= W) ? (j + w) : W;
                        float value = map[mi, mj];
                        // Цикл по высоте ядра
                        for (int k = i; k < th; k++)
                        {
                            // Цикл по ширине ядра
                            for (int l = j; l < tw; l++)
                            {
                                value += matrixes[m][k, l] * filter[m, k - i, l - j];
                            }
                        }
                        if (m == matrixes.Length - 1) value = activation(value);
                        map[mi, mj] = value;
                        unactivatedMap[mi, mj] = value;
                    }
                }
            }

            return new Pair<float[,], float[,]>(map, unactivatedMap);
        }

        // Свертка с same padding (выход - карта то же размера)
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Pair<float[,], float[,]>
            SamePaddingConvolution(float[][,] matrixes, float[,,] filter, int strides, ActivationFunction activation)
        {
            int H = matrixes[0].GetLength(0), h = filter.GetLength(1);
            int W = matrixes[0].GetLength(1), w = filter.GetLength(2);
            int dH = h - 1;
            int dW = w - 1;
            if ((dH % 2 != 0) || (dW % 2 != 0))
            {
                throw new ArgumentException("Размеры фильтра для Same padding должны быть нечетными");
            }
            int mh = (int)Math.Ceiling(H / (float)strides);
            int mw = (int)Math.Ceiling(W / (float)strides);
            float[,] map = new float[mh, mw];
            float[,] unactivatedMap = new float[H, W];
            dH /= 2;
            dW /= 2;

            for (int m = 0; m < matrixes.Length; m++)
            {
                for (int i = -dH, mi = 0; mi < mh; i += strides, mi++)
                {
                    for (int j = -dW, mj = 0; mj < mw; j += strides, mj++)
                    {
                        int th = (i + h <= H) ? (i + h) : H;
                        int tw = (j + w <= W) ? (j + w) : W;
                        float value = map[mi, mj];
                        for (int k = (i >= 0) ? i : 0; k < th; k++)
                        {
                            for (int l = (j >= 0) ? j : 0; l < tw; l++)
                            {
                                value += matrixes[m][k, l] * filter[m, k - i, l - j];
                            }
                        }
                        if (m == matrixes.Length - 1) value = activation(value);
                        map[mi, mj] = value;
                        unactivatedMap[mi, mj] = value;
                    }
                }
            }

            return new Pair<float[,], float[,]>(map, unactivatedMap);
        }

        // Свертка с full padding (выход - карта большего размера)
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Pair<float[,], float[,]>
            FullPaddingConvolution(float[][,] matrixes, float[,,] filter, int strides, ActivationFunction activation)
        {
            int H = matrixes[0].GetLength(0), h = filter.GetLength(1), mh = H + h - 1;
            int W = matrixes[0].GetLength(1), w = filter.GetLength(2), mw = W + w - 1;
            float[,] map = new float[mh, mw];
            float[,] unactivatedMap = new float[mh, mw];
            int dH = h - 1;
            int dW = w - 1;

            for (int m = 0; m < matrixes.Length; m++)
            {
                for (int i = 0, mi = -dH; i < mh; i++, mi++)
                {
                    for (int j = 0, mj = -dW; j < mw; j++, mj++)
                    {
                        float value = map[i, j];
                        for (int k = mi, ki = 0; k < mi + h; k++, ki++)
                        {
                            for (int l = mj, kj = 0; l < mj + w; l++, kj++)
                            {
                                value += (k < 0 || l < 0 || k >= H || l >= W) ?
                                    0 : (matrixes[m][k, l] * filter[m, ki, kj]);
                            }
                        }
                        if (m == matrixes.Length - 1) value = activation(value);
                        map[i, j] = value;
                        unactivatedMap[i, j] = value;
                    }
                }
            }

            return new Pair<float[,], float[,]>(map, unactivatedMap);
        }
    }
}

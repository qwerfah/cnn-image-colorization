
using System;
using System.Runtime.CompilerServices;

namespace CNN.Convolution.Layers.Functions
{
    public delegate float[,,] PoolingFunction(float[,,] data, int scaleX, int scaleY);

    public enum PoolingType
    {
        MaxPooling,
        AveragePooling,
        SumPooling,
        LSqrSumPooling,
        None
    }

    public static class PoolingFunctions
    {
        public static PoolingFunction[] Functions =
        {
            MaxPooling,
            AveragePooling,
            SumPooling,
            LSqrSumPooling,
        };

        // Максимум из указанной области
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float[,,] MaxPooling(float[,,] data, int scaleX, int scaleY)
        {
            int depth = data.GetLength(0);
            int height = data.GetLength(1);
            int width = data.GetLength(2);
            int dh = height / scaleY, dw = width / scaleX;
            float[,,] maps = new float[data.Length, height, width];

            for (int d = 0; d < depth; d++)
            {
                for (int i = 0; i < height; i += scaleY)
                {
                    for (int j = 0; j < width; j += scaleX)
                    {
                        float max = data[d, i, j];
                        for (int y = i; y < i + scaleY; y++)
                        {
                            for (int x = j; x < j + scaleX; x++)
                            {
                                float value = data[d, x, y];
                                if (value > max) max = value;
                            }
                        }
                        maps[d, i / scaleY, j / scaleX] = max;
                    }
                }
            }

            return maps;
        }

        // Среднее значение элементов указанной области
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float[,,] AveragePooling(float[,,] data, int scaleX, int scaleY)
        {
            int depth = data.GetLength(0);
            int height = data.GetLength(1);
            int width = data.GetLength(2);
            int dh = height / scaleY, dw = width / scaleX;
            float[,,] maps = new float[data.Length, height, width];
            float del = scaleX * scaleY;

            for (int d = 0; d < depth; d++)
            {
                for (int i = 0; i < height; i += scaleY)
                {
                    for (int j = 0; j < width; j += scaleX)
                    {
                        float sum = 0;
                        for (int y = i; y < i + scaleY; y++)
                        {
                            for (int x = j; x < j + scaleX; x++)
                            {
                                sum += data[d, x, y];
                            }
                        }
                        maps[d, i / scaleY, j / scaleX] = sum / del;
                    }
                }
            }

            return maps;
        }

        // Сумма элементов указанной области
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float[,,] SumPooling(float[,,] data, int scaleX, int scaleY)
        {
            int depth = data.GetLength(0);
            int height = data.GetLength(1);
            int width = data.GetLength(2);
            int dh = height / scaleY, dw = width / scaleX;
            float[,,] maps = new float[data.Length, height, width];

            for (int d = 0; d < depth; d++)
            {
                for (int i = 0; i < height; i += scaleY)
                {
                    for (int j = 0; j < width; j += scaleX)
                    {
                        float sum = 0;
                        for (int y = i; y < i + scaleY; y++)
                        {
                            for (int x = j; x < j + scaleX; x++)
                            {
                                sum += data[d, x, y];
                            }
                        }
                        maps[d, i / scaleY, j / scaleX] = sum;
                    }
                }
            }

            return maps;
        }

        // Норма L^2 (корень из суммы квадратов)
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float[,,] LSqrSumPooling(float[,,] data, int scaleX, int scaleY)
        {
            int depth = data.GetLength(0);
            int height = data.GetLength(1);
            int width = data.GetLength(2);
            int dh = height / scaleY, dw = width / scaleX;
            float[,,] maps = new float[data.Length, height, width];

            for (int d = 0; d < depth; d++)
            {
                for (int i = 0; i < height; i += scaleY)
                {
                    for (int j = 0; j < width; j += scaleX)
                    {
                        float sum = 0;
                        for (int y = i; y < i + scaleY; y++)
                        {
                            for (int x = j; x < j + scaleX; x++)
                            {
                                sum += data[d, x, y] * data[d, x, y];
                            }
                        }
                        maps[d, i / scaleY, j / scaleX] = (float)Math.Sqrt(sum);
                    }
                }
            }

            return maps;
        }
    }
}

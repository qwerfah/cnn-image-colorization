namespace CNN.Convolution.Layers.Functions
{
    public delegate float[,,] UpsamplingFunction(float[,,] matrix, int xScale, int yScale);

    public enum UpsamplingType
    {
        BilinearUpsampling,
        NearestNeighborUpsampling
    }

    public static class UpsamplingFunctions
    {
        public static UpsamplingFunction[] Functions =
        {
            BilinearUpsampling,
            NearestNeighborUpsampling
        };

        // Upsampling с билинейной интерполяцией
        public static float[,,] BilinearUpsampling(float[,,] data, int xScale, int yScale)
        {
            int height = data.GetLength(1) * yScale;
            int width = data.GetLength(2) * xScale;
            float[,,] upscaled = new float[data.Length, height, width];

            for (int f = 0; f < data.Length; f++)
            {
                for (int i = 0, mi = 0; i < height; i += yScale, mi++)
                {
                    for (int j = 0, mj = 0; j < width; j += xScale, mj++)
                    {
                        float temp = data[f, mi, mj];
                        for (int ys = i; ys < i + yScale; ys++)
                        {
                            for (int xs = j; xs < j + xScale; xs++)
                            {
                                upscaled[f, ys, xs] = temp;
                            }
                        }
                    }
                }
            }

            return upscaled;
        }

        // Upsampling с интерполяцией методом ближайших соседей
        public static float[,,] NearestNeighborUpsampling(float[,,] data, int xScale, int yScale)
        {
            int depth = data.GetLength(0);
            int height = data.GetLength(1) * yScale;
            int width = data.GetLength(2) * xScale;
            float[,,] upscaled = new float[depth, height, width];

            for (int f = 0; f < depth; f++)
            {
                for (int i = 0, mi = 0; i < height; i += yScale, mi++)
                {
                    for (int j = 0, mj = 0; j < width; j += xScale, mj++)
                    {
                        float temp = data[f, mi, mj];
                        for (int ys = i; ys < i + yScale; ys++)
                        {
                            for (int xs = j; xs < j + xScale; xs++)
                            {
                                upscaled[f, ys, xs] = temp;
                            }
                        }
                    }
                }
            }

            return upscaled;
        }
    }
}

using CNN.AuxiliaryStructures;
using System;
using System.Drawing;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace CNN.DataProcessing
{
    public static partial class IOProcessing
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Bitmap LoadImageFromFile(string path)
        {
            if (path == null) return null;
            Bitmap map = new Bitmap(path);
            if (map == null) return null;

            return map;
        }

        // Перевод bitmap в RGB в трехмерный тензор модели с яркостью
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Pair<float[,,], float[,,]> RgbToAny(Bitmap rgb,
            RgbToModel rgbToModel)
        {
            float[,,] y = new float[1, rgb.Height, rgb.Width];
            float[,,] uv = new float[2, rgb.Height, rgb.Width];

            for (int i = 0; i < rgb.Height; i++)
            {
                for (int j = 0; j < rgb.Width; j++)
                {
                    float[] color = rgbToModel(rgb.GetPixel(j, i));
                    y[0, i, j] = color[0];
                    uv[0, i, j] = color[1];
                    uv[1, i, j] = color[2];
                }
            }

            return new Pair<float[,,], float[,,]>(y, uv);
        }

        // Перевод bitmap в RGB в трехмерный тензор модели с яркостью
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Pair<float[,,], float[,,]> RgbToSample(Bitmap rgb)
        {
            float[,,] input = new float[1, rgb.Height, rgb.Width];
            float[,,] output = new float[3, rgb.Height, rgb.Width];

            for (int i = 0; i < rgb.Height; i++)
            {
                for (int j = 0; j < rgb.Width; j++)
                {
                    System.Drawing.Color color = rgb.GetPixel(j, i);
                    float r = color.R;
                    float g = color.G;
                    float b = color.B;

                    input[0, i, j] = (r + g + b) / 3.0f / 255.0f;
                    output[0, i, j] = r / 255.0f;
                    output[1, i, j] = g / 255.0f;
                    output[2, i, j] = b / 255.0f;
                }
            }

            return new Pair<float[,,], float[,,]>(input, output);
        }

        // Перевод тензора в RGB bitmap
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Bitmap AnyToRgb(Pair<float[,,], float[,,]> data,
            ModelToRgb modelToRgb)
        {
            int height = data.Item2.GetLength(1);
            int width = data.Item2.GetLength(2);

            Bitmap bitmap = new Bitmap(width, height);

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    float[] color = new float[3]
                    {
                        data.Item1[0, i, j],
                        data.Item2[0, i, j],
                        data.Item2[1, i, j],
                    };
                    bitmap.SetPixel(j, i, modelToRgb(color));
                }
            }

            return bitmap;
        }

        // Перевод тензора в RGB bitmap
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Bitmap SampleToRgb(float[,,] data)
        {
            int height = data.GetLength(1);
            int width = data.GetLength(2);

            Bitmap bitmap = new Bitmap(width, height);

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    bitmap.SetPixel(j, i,
                        System.Drawing.Color.FromArgb((int)(data[0, i, j] * 255.0),
                        (int)(data[1, i, j] * 255.0), (int)(data[2, i, j] * 255.0)));
                }
            }

            return bitmap;
        }

        public static BitmapSource ConvertBitmapToSource(Bitmap bitmap)
        {
            /*
            var bitmapData = bitmap.LockBits(
                new System.Drawing.Rectangle(0, 0, bitmap.Width, bitmap.Height),
                System.Drawing.Imaging.ImageLockMode.ReadOnly, bitmap.PixelFormat);

            var bitmapSource = BitmapSource.Create(
                bitmapData.Width, bitmapData.Height,
                bitmap.HorizontalResolution, bitmap.VerticalResolution,
                ConvertPixelFormat(bitmap.PixelFormat), null,
                bitmapData.Scan0, bitmapData.Stride * bitmapData.Height, bitmapData.Stride);

            bitmap.UnlockBits(bitmapData);
            return bitmapSource;
            */
            var hBitmap = bitmap.GetHbitmap();
            var drawable = System.Windows.Interop.Imaging.CreateBitmapSourceFromHBitmap(
                  hBitmap,
                  IntPtr.Zero,
                  Int32Rect.Empty,
                  BitmapSizeOptions.FromEmptyOptions());
            return drawable;
        }

        private static PixelFormat ConvertPixelFormat(System.Drawing.Imaging.PixelFormat sourceFormat)
        {
            switch (sourceFormat)
            {
                case System.Drawing.Imaging.PixelFormat.Format24bppRgb:
                    return PixelFormats.Bgr24;

                case System.Drawing.Imaging.PixelFormat.Format32bppArgb:
                    return PixelFormats.Bgra32;

                case System.Drawing.Imaging.PixelFormat.Format32bppRgb:
                    return PixelFormats.Bgr32;
            }
            return new System.Windows.Media.PixelFormat();
        }
    }
}

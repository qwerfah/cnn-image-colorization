using CNN.AuxiliaryStructures;
using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;

namespace CNN.DataProcessing
{
    /// <summary>
    /// Предоставляет функции для предобработки и постобработки изображений.
    /// </summary>
    public static partial class IOProcessing
    {
        /// <summary>
        /// Перевод цвета RGB в модель CIE LAB.
        /// </summary>
        /// <param name="r">Канал красного цвета.</param>
        /// <param name="g">Канал зеленого цвета.</param>
        /// <param name="b">Канал синего цвета.</param>
        /// <returns></returns>
        private static float[] RgbToLab(byte r, byte g, byte b)
        {
            float var_B = b / 255.0f;
            float var_G = g / 255.0f;
            float var_R = r / 255.0f;

            if (var_R > 0.04045) var_R = (float)Math.Pow(((var_R + 0.055f) / 1.055f), 2.4f);
            else var_R = var_R / 12.92f;
            if (var_G > 0.04045) var_G = (float)Math.Pow(((var_G + 0.055f) / 1.055f), 2.4f);
            else var_G = var_G / 12.92f;
            if (var_B > 0.04045) var_B = (float)Math.Pow(((var_B + 0.055f) / 1.055f), 2.4f);
            else var_B = var_B / 12.92f;

            var_R *= 100.0f;
            var_G *= 100.0f;
            var_B *= 100.0f;

            float X = var_R * 0.4124f + var_G * 0.3576f + var_B * 0.1805f;
            float Y = var_R * 0.2126f + var_G * 0.7152f + var_B * 0.0722f;
            float Z = var_R * 0.0193f + var_G * 0.1192f + var_B * 0.9505f;

            float var_X = X / 95.047f;
            float var_Y = Y / 100f;
            float var_Z = Z / 108.883f;

            if (var_X > 0.008856f) var_X = (float)Math.Pow(var_X, (1.0f / 3.0f));
            else var_X = (7.787f * var_X) + (16.0f / 116.0f);
            if (var_Y > 0.008856f) var_Y = (float)Math.Pow(var_Y, (1.0f / 3.0f));
            else var_Y = (7.787f * var_Y) + (16.0f / 116.0f);
            if (var_Z > 0.008856f) var_Z = (float)Math.Pow(var_Z, (1.0f / 3.0f));
            else var_Z = (7.787f * var_Z) + (16.0f / 116.0f);

            float l_s = (116.0f * var_Y) - 16.0f;
            float a_s = 500.0f * (var_X - var_Y);
            float b_s = 200.0f * (var_Y - var_Z);

            return new float[3] { l_s, a_s, b_s };
            
        }

        ///<summary>Преобразует изображение в структуру Image.</summary>
        ///<param name="Image">Изображение.</param>
        public static unsafe Image BitmapToImage(Bitmap image)
        {
            image = new Bitmap(image, Settings.ImageWidth, Settings.ImageHeight);
            Tensor3D input = new Tensor3D(image.Width, image.Height, 1);
            var BD = image.LockBits(new Rectangle(0, 0, image.Width, image.Height), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            int w = image.Width;

            for (int y = 0; y < image.Height; y++)
            {
                var Addr = (byte*)(BD.Scan0.ToInt64() + BD.Stride * y);

                for (int x = 0; x < w; x++)
                {
                    var B = *Addr;
                    Addr += 1;
                    var G = *Addr;
                    Addr += 1;
                    var R = *Addr;
                    Addr += 1;
                    float[] lab = RgbToLab(R, G, B);

                    input.Set(x, y, 0, lab[0] - 50.0f);
                }
            }
            image.UnlockBits(BD);

            image = new Bitmap(image, Settings.ImageWidth / 4, Settings.ImageHeight / 4);
            Tensor3D output = new Tensor3D(image.Width, image.Height, 2);
            BD = image.LockBits(new Rectangle(0, 0, image.Width, image.Height), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            w = image.Width;

            for (int y = 0; y < image.Height; y++)
            {
                var Addr = (byte*)(BD.Scan0.ToInt64() + BD.Stride * y);

                for (int x = 0; x < w; x++)
                {
                    var B = *Addr;
                    Addr += 1;
                    var G = *Addr;
                    Addr += 1;
                    var R = *Addr;
                    Addr += 1;
                    float[] lab = RgbToLab(R, G, B);

                    output.Set(x, y, 0, lab[1]);
                    output.Set(x, y, 1, lab[2]);
                }
            }
            image.UnlockBits(BD);

            return new Image { Input = input, Output = output };
        }

        /// <summary>
        /// Перевод цвета CIE LAB в пространство RGB.
        /// </summary>
        /// <param name="l_s">Компонента яркости.</param>
        /// <param name="a_s">Цветоразностная компонента A.</param>
        /// <param name="b_s">Цветоразностная компонента B.</param>
        /// <param name="R">Канал красного цвета.</param>
        /// <param name="G">Канал зеленого цвета.</param>
        /// <param name="B">Канал синего цвета.</param>
        private static void LabToRgb(float l_s, float a_s, float b_s, ref float R, ref float G, ref float B)
        {
            float var_Y = (l_s + 16.0f) / 116.0f;
            float var_X = a_s / 500.0f + var_Y;
            float var_Z = var_Y - b_s / 200.0f;

            if (Math.Pow(var_Y, 3.0f) > 0.008856f) var_Y = (float)Math.Pow(var_Y, 3.0f);
            else var_Y = (var_Y - 16.0f / 116.0f) / 7.787f;
            if (Math.Pow(var_X, 3.0f) > 0.008856) var_X = (float)Math.Pow(var_X, 3.0f);
            else var_X = (var_X - 16.0f / 116.0f) / 7.787f;
            if (Math.Pow(var_Z, 3.0f) > 0.008856f) var_Z = (float)Math.Pow(var_Z, 3.0f);
            else var_Z = (var_Z - 16.0f / 116.0f) / 7.787f;

            float X = 95.047f * var_X;
            float Y = 100.000f * var_Y;
            float Z = 108.883f * var_Z;

            var_X = X / 100.0f;
            var_Y = Y / 100.0f; 
            var_Z = Z / 100.0f;

            float var_R = var_X * 3.2406f + var_Y * -1.5372f + var_Z * -0.4986f;
            float var_G = var_X * -0.9689f + var_Y * 1.8758f + var_Z * 0.0415f;
            float var_B = var_X * 0.0557f + var_Y * -0.2040f + var_Z * 1.0570f;

            if (var_R > 0.0031308f) var_R = 1.055f * (float)Math.Pow(var_R, (1.0f / 2.4f)) - 0.055f;
            else var_R = 12.92f * var_R;
            if (var_G > 0.0031308f) var_G = 1.055f * (float)Math.Pow(var_G, (1.0f / 2.4f)) - 0.055f;
            else var_G = 12.92f * var_G;
            if (var_B > 0.0031308f) var_B = 1.055f * (float)Math.Pow(var_B, (1.0f / 2.4f)) - 0.055f;
            else var_B = 12.92f * var_B;

            R = var_R * 255.0f;
            G = var_G * 255.0f;
            B = var_B * 255.0f;
        }

        ///<summary>Преобразует тензор в изображение.</summary>
        public static unsafe Pair<Bitmap, Bitmap> TensorToImage(Bitmap l, Tensor3D ab)
        {
            var tmp = ResizeImage(l, ab.Width, ab.Height);
            var BD = tmp.LockBits(new Rectangle(0, 0, tmp.Width, tmp.Height), ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);

            for (int _y = 0; _y < tmp.Height; _y++)
            {
                byte* Addr = (byte*)(BD.Scan0.ToInt64() + BD.Stride * _y);
                for (int _x = 0; _x < tmp.Width; _x++)
                {
                    float r = *(Addr + 2);
                    float g = *(Addr + 1);
                    float b = *(Addr);
                    LabToRgb(RgbToLab((byte)r, (byte)g, (byte)b)[0], 
                        ab.Get(_x, _y, 0), ab.Get(_x, _y, 1), ref r, ref g, ref b);
                    *(Addr) = (byte)(Math.Min(255.0f, Math.Max(0.0f, b)));
                    Addr += 1;
                    *(Addr) = (byte)(Math.Min(255.0f, Math.Max(0.0f, g)));
                    Addr += 1;
                    *(Addr) = (byte)(Math.Min(255.0f, Math.Max(0.0f, r)));
                    Addr += 1;
                }
            }
            tmp.UnlockBits(BD);

            Bitmap original = tmp;
            tmp = ResizeImage(tmp, l.Width, l.Height);
            var BD_ab = tmp.LockBits(new Rectangle(0, 0, tmp.Width, tmp.Height), ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);
            var BD_l = l.LockBits(new Rectangle(0, 0, l.Width, l.Height), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            
            for (int y = 0; y < l.Height; y++)
            {
                byte* Addr_ab = (byte*)(BD_ab.Scan0.ToInt64() + BD_ab.Stride * y);
                byte* Addr_l = (byte*)(BD_l.Scan0.ToInt64() + BD_l.Stride * y);
                for (int x = 0; x < l.Width; x++)
                {
                    var Y = *(Addr_l + 2) * 0.257 + *(Addr_l + 1) * 0.504 + *Addr_l * 0.098 + 16.0;
                    var U = *(Addr_ab + 2) * -0.148 + *(Addr_ab + 1) * -0.291 + *Addr_ab * 0.439 + 128.0;
                    var V = *(Addr_ab + 2) * 0.439 + *(Addr_ab + 1) * -0.368 + *Addr_ab * -0.071 + 128.0;

                    var R = 1.164 * (Y - 16.0) + 1.596 * (V - 128.0);
                    var G = 1.164 * (Y - 16.0) - 0.391 * (U - 128.0) - 0.813 * (V - 128.0);
                    var B = 1.164 * (Y - 16.0) + 2.018 * (U - 128.0);

                    R = Math.Min(Math.Max(R, 0.0), 255.0);
                    G = Math.Min(Math.Max(G, 0.0), 255.0);
                    B = Math.Min(Math.Max(B, 0.0), 255.0);

                    *Addr_ab = (byte)B;
                    Addr_ab += 1;
                    *Addr_ab = (byte)G;
                    Addr_ab += 1;
                    *Addr_ab = (byte)R;
                    Addr_ab += 1;
                    Addr_l += 3;
                }
            }
            tmp.UnlockBits(BD_ab);
            l.UnlockBits(BD_l);

            return new Pair<Bitmap, Bitmap>(original, tmp);
        }

        /// <summary>
        /// Масштабирование изображения до указанного разрешения.
        /// </summary>
        /// <param name="image">Исходное изображение.</param>
        /// <param name="width">Целевая ширина изображения.</param>
        /// <param name="height">Целевая высота изображения.</param>
        /// <returns></returns>
        public static Bitmap ResizeImage(Bitmap image, int width, int height)
        {
            var destRect = new Rectangle(0, 0, width, height);
            var destImage = new Bitmap(width, height);

            destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            using (var graphics = Graphics.FromImage(destImage))
            {
                graphics.CompositingMode = CompositingMode.SourceCopy;
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

                using (var wrapMode = new ImageAttributes())
                {
                    wrapMode.SetWrapMode(WrapMode.TileFlipXY);
                    graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);
                }
            }

            return destImage;
        }
    }
}

using System;
using System.Drawing;
using System.Runtime.CompilerServices;

namespace CNN.DataProcessing
{
    public enum ColorModel
    {
        Lab = 0,
        Yuv,
        Yiq
    }

    public delegate float[] RgbToModel(Color rgb);
    public delegate Color ModelToRgb(float[] model);

    public static class ColorConversion
    {
        public static RgbToModel[] FromRgb =
        {
            RgbToLab,
            RgbToYuv,
            RgbToYiq,
        };

        public static ModelToRgb[] ToRgb =
        {
            LabToRgb,
            YuvToRgb,
            YiqToRgb,
        };
        // BT.601
        // Kr = 0.299f;
        // Kb = 0.114f;
        // BT.709
        // Kr = 0.2126f;
        // Kb = 0.0722f;
        // BT.2020
        public static float Kr { get; } = 0.2627f;
        public static float Kb { get; } = 0.0593f;

        private static float _yuvDelta = 1.0f - Kr - Kb;

        public static float[] D65 = new float[3] { 0.9505f, 1.0f, 1.0890f };
        private static float _xyzConst = 16f / 116f;
        private static float _xyzToLabPow = 1f / 3f;
        private static float _xyzToRgbPow = 1.0f / 2.4f;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float[] RgbToYuv(Color rgb)
        {
            float[] yuv = new float[3];
            // Y = Kr * R + (1f - Kr - Kb) * G + Kb * B
            yuv[0] = Kr * rgb.R + (1f - Kr - Kb) * rgb.G + Kb * rgb.B;
            // U = B - Y
            yuv[1] = rgb.B - yuv[0];
            // V = R - Y
            yuv[2] = rgb.R - yuv[0];

            return yuv;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Color YuvToRgb(float[] yuv)
        {
            float y = yuv[0], u = yuv[1], v = yuv[2];
            int r = (int)(y + v);
            int g = (int)(y - (Kr * v + Kb * u) / _yuvDelta);
            int b = (int)(y + u);

            return Color.FromArgb(r, g, b);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float[] RgbToYiq(Color rgb)
        {
            float[] yiq = new float[3];

            float r = (float)rgb.R / 255.0f;
            float g = (float)rgb.G / 255.0f;
            float b = (float)rgb.B / 255.0f;

            // Y = 0.299 * R + 0.587 * G + 0.114 * B
            yiq[0] = 0.299f * r + 0.587f * g + 0.114f * b;
            // I = 0.596 * R - 0.274 * G - 0.322 * B
            yiq[1] = 0.596f * r - 0.274f * g - 0.322f * b;
            // Q = 0.211 * R - 0.522 * G + 0.311 * B
            yiq[2] = 0.211f * r - 0.522f * g + 0.311f * b;

            return yiq;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Color YiqToRgb(float[] yiq)
        {
            float y = yiq[0], i = yiq[1], q = yiq[2];
            // R = Y + 0.956 * I + 0.623 * Q
            float r = y + 0.956f * i + 0.623f * q;
            // G = Y - 0.272 * I - 0.648 * Q
            float g = y - 0.272f * i - 0.648f * q;
            // B = Y - 1.105 * I + 1.705 * Q
            float b = y - 1.105f * i + 1.705f * q;

            return Color.FromArgb((int)(r * 255.0), (int)(g * 255.0), (int)(b * 255.0));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float[] RgbToXyz(Color rgb)
        {
            float r = (rgb.R / 255f);
            float g = (rgb.G / 255f);
            float b = (rgb.B / 255f);

            if (r > 0.04045) r = (float)Math.Pow(((r + 0.055f) / 1.055f), 2.4);
            else r /= 12.92f;
            if (g > 0.04045) g = (float)Math.Pow(((g + 0.055f) / 1.055f), 2.4);
            else g /= 12.92f;
            if (b > 0.04045) b = (float)Math.Pow(((b + 0.055f) / 1.055f), 2.4);
            else b /= 12.92f;

            r *= 100f;
            g *= 100f;
            b *= 100f;

            float x = r * 0.4124f + g * 0.3576f + b * 0.1805f;
            float y = r * 0.2126f + g * 0.7152f + b * 0.0722f;
            float z = r * 0.0193f + g * 0.1192f + b * 0.9505f;

            return new float[3] { x, y, z };
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Color XyzToRgb(float[] xyz)
        {
            float x = xyz[0] / 100f;
            float y = xyz[1] / 100f;
            float z = xyz[2] / 100f;

            float r = x * 3.2406f + y * -1.5372f + z * -0.4986f;
            float g = x * -0.9689f + y * 1.8758f + z * 0.0415f;
            float b = x * 0.0557f + y * -0.2040f + z * 1.0570f;

            if (r > 0.0031308f) r = 1.055f * (float)Math.Pow(r, _xyzToRgbPow) - 0.055f;
            else r = 12.92f * r;
            if (g > 0.0031308f) g = 1.055f * (float)Math.Pow(g, _xyzToRgbPow) - 0.055f;
            else g = 12.92f * g;
            if (b > 0.0031308f) b = 1.055f * (float)Math.Pow(b, _xyzToRgbPow) - 0.055f;
            else b = 12.92f * b;

            r *= 255.0f;
            g *= 255.0f;
            b *= 255.0f;

            if (r > 255) r = 255; if (r < 0) r = 0;
            if (g > 255) g = 255; if (g < 0) g = 0;
            if (b > 255) b = 255; if (b < 0) b = 0;

            return Color.FromArgb((int)(r), (int)(g), (int)(b));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float[] XyzToLab(float[] xyz)
        {
            float x = xyz[0] / D65[0];
            float y = xyz[1] / D65[1];
            float z = xyz[2] / D65[2];

            if (x > 0.008856f) x = (float)Math.Pow(x, _xyzToLabPow);
            else x = 7.787f * x + _xyzConst;
            if (y > 0.008856f) y = (float)Math.Pow(y, _xyzToLabPow);
            else y = 7.787f * y + _xyzConst;
            if (z > 0.008856f) z = (float)Math.Pow(z, _xyzToLabPow);
            else z = 7.787f * z + _xyzConst;

            float l = 116f * y - 16f;
            float a = 500f * (x - y);
            float b = 200f * (y - z);

            return new float[3] { l, a, b };
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float[] LabToXyz(float[] lab)
        {
            float var_Y = (lab[0] * 100.0f + 16f) / 116f;
            float var_X = lab[1] * 128.0f / 500f + var_Y;
            float var_Z = var_Y - lab[2] * 128.0f / 200f;

            float pow_X = (float)Math.Pow(var_X, 3);
            float pow_Y = (float)Math.Pow(var_Y, 3);
            float pow_Z = (float)Math.Pow(var_Z, 3);

            var_X = (pow_X > 0.008856f) ? pow_X : (var_X - _xyzConst) / 7.787f;
            var_Y = (pow_Y > 0.008856f) ? pow_Y : (var_Y - _xyzConst) / 7.787f;
            var_Z = (pow_Z > 0.008856f) ? pow_Z : (var_Z - _xyzConst) / 7.787f;

            float x = var_X * D65[0];
            float y = var_Y * D65[1];
            float z = var_Z * D65[2];

            return new float[3] { x, y, z };
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float[] RgbToLab(Color rgb)
        {
            return XyzToLab(RgbToXyz(rgb));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Color LabToRgb(float[] lab)
        {
            return XyzToRgb(LabToXyz(lab));
        }
    }
}

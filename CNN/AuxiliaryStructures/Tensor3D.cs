using System;

namespace CNN.AuxiliaryStructures
{
    [Serializable]
    public struct Tensor3D
    {
        public float[] W { get; set; }

        public int Width { get; set; }
        public int Height { get; set; }
        public int Depth { get; set; }

        public Tensor3D(int w, int h, int d)
        {
            W = new float[w * h * d];
            Width = w;
            Height = h;
            Depth = d;
        }

        public float Get(int x, int y, int z)
        {
            return W[((Width * y) + x) * Depth + z];
        }

        public void Set(int x, int y, int z, float v)
        {
            W[((Width * y) + x) * Depth + z] = v;
        }

        public void Add(int x, int y, int z, float v)
        {
            W[((Width * y) + x) * Depth + z] += v;
        }

        public void Multiply(int x, int y, int z, float v)
        {
            W[((Width * y) + x) * Depth + z] *= v;
        }
    }
}

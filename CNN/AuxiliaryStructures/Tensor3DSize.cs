using System;

namespace CNN.AuxiliaryStructures
{
    [Serializable]
    public struct Tensor3DSize
    {
        public int Depth { get; }
        public int Height { get; }
        public int Width { get; }

        public Tensor3DSize(int depth, int height, int width)
        {
            Depth = depth;
            Height = height;
            Width = width;
        }
    }
}

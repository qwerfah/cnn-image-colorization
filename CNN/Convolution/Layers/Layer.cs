using CNN.AuxiliaryStructures;
using System;

namespace CNN.Convolution.Layers
{
    ///<summary>Тип слоя сверточной нейронной сети.</summary>
    public enum LayerType
    {
        Convolution,
        Deconvolution,
        Subsampling,
        Upsampling,
        Activation,
        BatchNorm,
        Softmax
    }

    ///<summary>Абстрактный класс слоя сверточной нейронной сети.</summary>
    [Serializable]
    public abstract class Layer
    {
        ///<summary>Размер входных данных.</summary>
        protected Tensor3DSize _inputSize;
        ///<summary>Размер выходных данных.</summary>
        protected Tensor3DSize _outputSize;

        ///<summary>Обработка тензора при прямом ходе по сети.</summary>
        public abstract Tensor3D Transfer(Tensor3D data);
        ///<summary>Метод обратного хода по сети.</summary>
        public abstract Tensor3D BackPropagation(Tensor3D nextLayerDeltas, Tensor3D input, Tensor3D output);
        ///<summary>Вычисление градиентов для выходного слоя.</summary>
        public abstract Tensor3D CalcOutputDeltas(Tensor3D result, Tensor3D correct);
        ///<summary>Вычисление гиперпараметров входных и выходных данных слоя.</summary>
        public abstract Tensor3DSize CalcSizes(Tensor3DSize inputSize);
    }
}

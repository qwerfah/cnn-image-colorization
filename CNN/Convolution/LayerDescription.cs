using CNN.Convolution.Layers;
using CNN.Convolution.Layers.Functions;

namespace CNN.Convolution
{
    ///<summary>Описание слоя нейронной сети.</summary>
    public class LayerDescription
    {
        ///<summary>Тип слоя.</summary>
        public LayerType LayerType { get; set; }
        ///<summary>Высота фильтра.</summary>
        public int KernelHeight { get; set; } = 0;
        ///<summary>Ширина фильтра.</summary>
        public int KernelWidth { get; set; } = 0;
        ///<summary>Глубина фильтра.</summary>
        public int FilterDepth { get; set; } = 0;
        ///<summary>Число фильтров.</summary>
        public int FilterCount { get; set; } = 0;
        ///<summary>Тип пулинга.</summary>
        public PoolingType PoolingType { get; set; } = PoolingType.None;
        ///<summary>Функция активации.</summary>
        public ActivationType ActivationType { get; set; } = ActivationType.None;
        ///<summary>Число страйдов (шаг фильтра).</summary>
        public int Strides { get; set; } = 0;
        ///<summary>Степерь разреженности фильтра.</summary>
        public int Dilation { get; set; } = 0;
    }
}

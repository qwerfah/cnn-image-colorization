using CNN.AuxiliaryStructures;

namespace CNN.DataProcessing
{
    /// <summary> Структура, хранящая изображение в цветовом пространстве CIE LAB. </summary>
    public struct Image
    {
        /// <summary> Компонента яркости (вход нейросети). </summary>
        public Tensor3D Input { get; set; }
        /// <summary> Компоненты цветов ab (выход нейросети). </summary>
        public Tensor3D Output { get; set; }
    }
}

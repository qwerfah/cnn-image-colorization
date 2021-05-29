using CNN.DataProcessing;

namespace CNN.AuxiliaryStructures
{
    /// <summary> Параметры обучения нейронной сети. </summary>
    public class TeachParams
    {
        /// <summary> Обучающая выборка. </summary>
        public Image[] Images { get; set; }
        /// <summary> Число эпох. </summary>
        public int Eras { get; set; }
        /// <summary> 
        /// Скорость обучения - масштабирующий
        /// коэффициент для шага вдоль градиента. 
        /// </summary>
        public float Rate { get; set; }
        /// <summary> 
        /// Импульс - коэффициент затухания для 
        /// скользящего среднего значения градиента. 
        /// </summary>
        public float Pulse { get; set; }
        /// <summary> 
        /// Момент - коэффициент затухания для 
        /// скользящего среднего квадрата градиента. 
        /// </summary>
        public float Momentum { get; set; }
        /// <summary> Размер пакета. </summary>
        public int BatchSize { get; set; }
    }
}

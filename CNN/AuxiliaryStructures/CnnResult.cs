using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN.AuxiliaryStructures
{
    /// <summary>
    /// Результаты обработки изображения нейронной сетью
    /// </summary>
    public struct CnnResult
    {
        /// <summary>
        /// Полученное на выходе сети изображение
        /// </summary>
        public Tensor3D Image { get; set; }
        /// <summary>
        /// Время обработки
        /// </summary>
        public long Time { get; set; }

        public CnnResult(Tensor3D image, long time)
        {
            Image = image;
            Time = time;
        }
    }
}

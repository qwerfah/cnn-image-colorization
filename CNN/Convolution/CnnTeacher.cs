using CNN.AuxiliaryStructures;
using CNN.DataProcessing;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Windows.Documents;

namespace CNN.Convolution
{
    /// <summary>
    /// Оптимизатор сверточной нейронной сети
    /// </summary>
    public class CnnTeacher
    {
        /// <summary>
        /// Ссылка на экземпляр сверточной сети, ассоциированной с оптимизатором.
        /// </summary>
        private readonly ConvolutionalNetwork _network;

        public CnnTeacher(ConvolutionalNetwork network)
        {
            _network = network;
        }

        /// <summary>
        /// Обучение сверточной нейронной сети.
        /// </summary>
        /// <param name="sender">Инициатор вызова события.</param>
        /// <param name="e">Данные для обработчика события</param>
        public void Teach(object sender, DoWorkEventArgs e)
        {
            TeachParams teachParams = e.Argument as TeachParams;
            if (teachParams == null)
            {
                throw new ArgumentNullException($"Оптимизатор: невалидный объект {nameof(teachParams)}");
            }
            if ((teachParams.Images== null) || (teachParams.Images.Length == 0))
            {
                throw new ArgumentException("Оптимизатор: поступила пустая обучающая выборка!");
            }
            if (teachParams.Eras < 1)
            {
                throw new ArgumentException("Оптимизатор: число эпох меньше 1!");
            }
            if (teachParams.BatchSize < 1)
            {
                throw new ArgumentException("Оптимизатор: размер пакета меньше 1!");
            }

            Image[] images = teachParams.Images.OrderBy(x => Settings.Random.Next()).ToArray();
            int total = 0;
            float step = 100.0f / (float)(teachParams.Images.Length * teachParams.Eras);
            float progress = 0;
            float b = (float)images.Length / (float)teachParams.BatchSize;
            int batches = (int)((Math.Floor(b) == b) ? b : (b + 1));

            // Цикл по эпохам
            for (int i = 0; i < teachParams.Eras; i++)
            {
                // Цикл по пакетам
                for (int j = 0; j < batches; j++)
                {
                    Image[] batch = GetBatchFromSample(images.Skip(teachParams.BatchSize * j), teachParams.BatchSize);
                    _network.BackPropagation(batch);
                    _network.UpdateWeights(teachParams.Rate, teachParams.Pulse, teachParams.Momentum);

                    progress += step; total++;
                    if (progress >= 2)
                    {
                        (sender as BackgroundWorker).ReportProgress((int)progress);
                        progress = 0;
                    }
                }
            }
        }

        /// <summary>
        /// Сформировать пакет заданного размера из элементов обучающей выборки.
        /// </summary>
        /// <param name="sample">Обучающая выборка.</param>
        /// <param name="batchSize">Размер пакета.</param>
        /// <returns>Массив (пакет) элементов обучающей выборки заданного размера</returns>
        private Image[] GetBatchFromSample(IEnumerable<Image> sample, int batchSize)
        {
            List<Image> batch = new List<Image>();

            while (batch.Count() < batchSize)
            {
                batch.AddRange(sample.Take(batchSize - batch.Count()).ToArray());
            }

            if (batch.Count() != batchSize) throw new ArgumentException("Shiiiiiiiiiiiiiiiiiiaaat");

            return batch.ToArray();
        }
    }
}

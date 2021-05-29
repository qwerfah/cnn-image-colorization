using CNN.AuxiliaryStructures;
using CNN.Convolution;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Documents;

namespace CNN.DataProcessing
{
    /// <summary>
    /// Предоставляет методы для параллельной загрузки данных, 
    /// а также выполняемые в отдельном потоке функции.
    /// </summary>
    public class ParallelDataProcessing
    {
        public ConvolutionalNetwork Network { get; } = null;
        public CnnTeacher Teacher { get; } = null;
        private static string[] _supportedExtensions = { "*.jpg", "*.png", "*.bmp", "*.gif", "*.tiff" };

        public ParallelDataProcessing(ConvolutionalNetwork network, CnnTeacher teacher)
        {
            Network = network;
            Teacher = teacher;
        }

        /// <summary>
        /// Загрузка целевого изображения и его обработка нейронной сетью.
        /// </summary>
        /// <param name="window">Экземпляр окна графического интерфейса.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Colorize(MainWindow window)
        {
            Bitmap inputImage = null;
            Pair<Bitmap, Bitmap> outputImage;
            Image image;

            try
            {
                inputImage = IOProcessing.LoadImageFromFile(window.ImagePathTextBox.Text);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Не удалось загрузить изображение по указанному пути!\n" +
                    $"Возможно, исходный файл имел неверный формат или поврежден.\n{ex.Message}");
                return;
            }

            try
            {
                image = IOProcessing.BitmapToImage(inputImage);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Не удалось сконвертировать изображение в формат для нейросети!\n{ex.Message}");
                return;
            }
            

            BackgroundWorker worker = new BackgroundWorker();

            window.ProgressBar.Value = 0;
            worker.WorkerReportsProgress = true;
            worker.DoWork += Network.Transfer;
            worker.ProgressChanged += (object sender, ProgressChangedEventArgs e) =>
            {
                window.ProgressBar.Value += e.ProgressPercentage;
            };
            worker.RunWorkerCompleted += (object sender, RunWorkerCompletedEventArgs e) =>
            {
                if (e.Error != null)
                {
                    MessageBox.Show(e.Error.Message);
                    window.ProgressBar.Value = 0;
                    window.Status.Content = "Ошибка";
                    window.IsEnabled = true;
                    return;
                }
                else
                {
                    window.ProgressBar.Value = 100;
                    window.Status.Content = "Готово";
                    window.IsEnabled = true;
                    CnnResult result;

                    try
                    {
                        result = (e.Result as CnnResult?).Value;
                        outputImage = IOProcessing.TensorToImage(inputImage, result.Image);
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show("Не удалось восстановить изображение " +
                            "на основе рассчитанных величин цветов!\n" + ex.Message);
                        return;
                    }

                    try
                    {
                        ImagesWindow imagesWindow = new ImagesWindow(inputImage, outputImage.Item1, 
                                                                     outputImage.Item2, result.Time);
                        imagesWindow.Show();
                    }
                    catch
                    {
                        MessageBox.Show("Не удалось отобразить полученный результат!\n" +
                            "Возможно, исходный файл имел неверный формат или поврежден.");
                    }
                }
            };

            window.Status.Content = "Колоризация";
            window.IsEnabled = false;

            worker.RunWorkerAsync(image.Input);
        }

        /// <summary>
        /// Загрузка тренировочной выборки и запуск обучения нейронной сети.
        /// </summary>
        /// <param name="path">Путь к обучающей выборке.</param>
        /// <param name="teachParams">Параметры обучения.</param>
        /// <param name="window">Экземпляр окна графического интерфейса.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void LoadAndTeach(string path, TeachParams teachParams, MainWindow window)
        {
            BackgroundWorker worker = new BackgroundWorker();

            window.ProgressBar.Value = 0;
            worker.WorkerReportsProgress = true;
            worker.DoWork += LoadImages;
            worker.ProgressChanged += (object sender, ProgressChangedEventArgs e) =>
            {
                window.ProgressBar.Value += e.ProgressPercentage;
            };
            worker.RunWorkerCompleted += (object sender, RunWorkerCompletedEventArgs e) =>
            {
                if (e.Error != null)
                {
                    MessageBox.Show(e.Error.Message);
                    window.ProgressBar.Value = 0;
                    window.Status.Content = "Ошибка";
                    window.IsEnabled = true;
                    return;
                }
                else
                {
                    window.ProgressBar.Value = 100;
                    teachParams.Images = e.Result as Image[];
                    Teach(teachParams, window);
                }
            };

            window.Status.Content = "Загрузка";
            window.IsEnabled = false;

            worker.RunWorkerAsync(new LoadParams { Path = path, Window = window });
        }

        /// <summary>
        /// Запуск обучения сети в отдельном потоке.
        /// </summary>
        /// <param name="teachParams">Параметры обучения.</param>
        /// <param name="window">Экземпляр окна графического интерфейса.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Teach(TeachParams teachParams, MainWindow window)
        {
            BackgroundWorker worker = new BackgroundWorker();

            window.ProgressBar.Value = 0;
            worker.WorkerReportsProgress = true;
            worker.DoWork += Teacher.Teach;
            worker.ProgressChanged += (object sender, ProgressChangedEventArgs e) =>
            {
                window.ProgressBar.Value += e.ProgressPercentage;
            };
            worker.RunWorkerCompleted += (object sender, RunWorkerCompletedEventArgs e) =>
            {
                if (e.Error != null)
                {
                    MessageBox.Show(e.Error.Message);
                    window.ProgressBar.Value = 0;
                    window.Status.Content = "Ошибка";
                    window.IsEnabled = true;
                    return;
                }
                else
                {
                    window.ProgressBar.Value = 100;
                    window.Status.Content = "Готово";
                    window.IsEnabled = true;
                }
            };

            window.Status.Content = "Обучение";
            window.IsEnabled = false;

            worker.RunWorkerAsync(teachParams);
        }

        /// <summary>
        /// Параллельная загрузка изображений в отдельном потоке.
        /// </summary>
        /// <param name="sender">Инициатор события вызова метода.</param>
        /// <param name="e">Аргументы.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void LoadImages(object sender, DoWorkEventArgs e)
        {
            LoadParams loadParams = e.Argument as LoadParams;
            string[] fileList = GetFileList(loadParams.Path);
            int count = fileList.GetLength(0) < 1000 ? fileList.GetLength(0) : 1000;
            Image[] images = new Image[count];
            float step = 100f / count;
            float curr = 0;

            Parallel.For(0, count, i =>
            {
                images[i] = IOProcessing.BitmapToImage(new Bitmap(fileList[i]));
                curr += step;
                if (curr >= 5)
                {
                    (sender as BackgroundWorker).ReportProgress((int)curr);
                    curr = 0;
                }

            });

            e.Result = images;
        }

        /// <summary>
        /// Формирование списка имен файлов изображений допустимых форматов по указанному пути.
        /// </summary>
        /// <param name="path">Путь к директории для поиска.</param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static string[] GetFileList(string path)
        {
            List<string> fileList = new List<string>();

            for (int i = 0; i < _supportedExtensions.Length; i++)
            {
                string[] files = Directory.GetFiles(path, _supportedExtensions[i], SearchOption.AllDirectories);
                fileList.AddRange(files);
            }

            return fileList.ToArray();
        }
    }
}

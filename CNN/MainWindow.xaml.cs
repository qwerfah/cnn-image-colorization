using CNN.AuxiliaryStructures;
using CNN.Convolution;
using CNN.Convolution.Layers;
using CNN.Convolution.Layers.Functions;
using CNN.DataProcessing;
using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

namespace CNN
{
    /// <summary>
    /// Логика взаимодействия для CnnWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public List<LayerDescription> Layers { get; set; }

        public ParallelDataProcessing ParallelDataProcessing { get; set; }

        public MainWindow()
        {
            InitializeComponent();
            LoadPredefinedArchitecture();
        }

        private void LoadPredefinedArchitecture()
        {
            int height = 0;
            int width = 0;

            try
            {
                width = int.Parse(WidthTextBox.Text);
                height = int.Parse(HeightTextBox.Text);
                Settings.ImageWidth = width;
                Settings.ImageHeight = height;
            }
            catch (Exception ex)
            {
                MessageBox.Show("Неверно заданы размеры изображения!\n" + ex.Message);
            }

            var arc = DescriptionToLayerConverter
                .GenerateArchitecture((CnnArchitecture)ArchitectureComboBox.SelectedItem);

            Layers = arc.Item1;
            LayersGrid.ItemsSource = Layers;
            ConvolutionalNetwork network = new ConvolutionalNetwork(new Tensor3DSize(1, height, width));
            network.AddLayers(arc.Item2);
            if ((CnnArchitecture)ArchitectureComboBox.SelectedItem == CnnArchitecture.VGG) network.LoadWeights();
            CnnTeacher teacher = new CnnTeacher(network);
            ParallelDataProcessing = new ParallelDataProcessing(network, teacher);
        }

        private void LoadArchitectureFromConstructor()
        {
            int height = 0;
            int width = 0;

            try
            {
                width = int.Parse(WidthTextBox.Text);
                height = int.Parse(HeightTextBox.Text);
                Settings.ImageWidth = width;
                Settings.ImageHeight = height;
            }
            catch (Exception ex)
            {
                MessageBox.Show("Неверно заданы размеры изображения!\n" + ex.Message);
                return;
            }

            ConvolutionalNetwork network = new ConvolutionalNetwork(new Tensor3DSize(1, height, width));

            try
            {
                for (int i = 0; i < Layers.Count(); i++)
                {
                    network.AddLayer(DescriptionToLayerConverter.DescriptionToLayer(Layers[i]));
                }
            }
            catch (Exception e)
            {
                MessageBox.Show("Некорректная архитектура!\n" + e.Message);
                return;
            }
            

            CnnTeacher teacher = new CnnTeacher(network);
            ParallelDataProcessing = new ParallelDataProcessing(network, teacher);
        }

        private void LoadArchitectureButton_Click(object sender, RoutedEventArgs e)
        {
            LoadPredefinedArchitecture();
        }

        private void RebuildButton_Click(object sender, RoutedEventArgs e)
        {
            LoadArchitectureFromConstructor();
        }

        private void UpdateButton_Click(object sender, RoutedEventArgs e)
        {
            int height = 0;
            int width = 0;

            try
            {
                width = int.Parse(WidthTextBox.Text);
                height = int.Parse(HeightTextBox.Text);
                Settings.ImageWidth = width;
                Settings.ImageHeight = height;
            }
            catch (Exception ex)
            {
                MessageBox.Show("Неверно заданы размеры изображения!\n" + ex.Message);
                return;
            }

            ParallelDataProcessing.Network.InputSize = new Tensor3DSize(1, height, width);
        }

        private void TeachButton_Click(object sender, RoutedEventArgs e)
        {
            int eras = 0;
            float rate = 0.0f, pulse = 0.0f, momentum = 0.0f;
            int batchSize = 0;

            try
            {
                eras = int.Parse(ErasTextBox.Text);
            }
            catch
            {
                MessageBox.Show("Неверно задано число эпох!");
                return;
            }
            try
            {
                rate = float.Parse(RateTextBox.Text);
                pulse = float.Parse(PulseTextBox.Text);
                momentum = float.Parse(MomentumTextBox.Text);
            }
            catch
            {
                MessageBox.Show("Неверно заданы параметры обучения!");
                return;
            }
            try
            {
                batchSize = int.Parse(BatchSizeTextBox.Text);
            }
            catch
            {
                MessageBox.Show("Неверно задан размер пакета!");
                return;
            }

            if (eras < 1) MessageBox.Show("Число эпох не может быть меньше 1!");

            try
            {
                TeachParams teachParams = new TeachParams 
                { 
                    Eras = eras, 
                    Rate = rate, 
                    Pulse = pulse, 
                    Momentum = momentum, 
                    BatchSize = batchSize 
                };
                ParallelDataProcessing
                    .LoadAndTeach(TeachPathTextBox.Text, teachParams, this);
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void ColorizeButton_Click(object sender, RoutedEventArgs e)
        {
            ParallelDataProcessing.Colorize(this);
        }

        private int DS_Count(string s)
        {
            string substr =
                System.Globalization.CultureInfo.CurrentCulture
                .NumberFormat.NumberDecimalSeparator[0].ToString();
            int count = (s.Length - s.Replace(substr, "").Length) / substr.Length;

            return count;
        }

        private void TextBox_PreviewTextInput(object sender, TextCompositionEventArgs e)
        {
            e.Handled =
                !((Char.IsDigit(e.Text, 0) ||
                ((e.Text == System.Globalization.CultureInfo.CurrentCulture
                .NumberFormat.NumberDecimalSeparator[0].ToString()) &&
                (DS_Count(((TextBox)sender).Text) < 1))));
        }

        private void OpenTeachDirectoryButton_Click(object sender, RoutedEventArgs e)
        {
            using (var fbd = new System.Windows.Forms.FolderBrowserDialog())
            {
                //fbd.RootFolder = Environment.SpecialFolder.ProgramFiles;
                System.Windows.Forms.DialogResult result = fbd.ShowDialog();

                if (result == System.Windows.Forms.DialogResult.OK && !string.IsNullOrWhiteSpace(fbd.SelectedPath))
                {
                    TeachPathTextBox.Text = fbd.SelectedPath;

                }
            }
        }

        private void OpenImageButton_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog dialog = new OpenFileDialog();
            dialog.InitialDirectory = ImagePathTextBox.Text;

            if (dialog.ShowDialog() == true)
            {
                ImagePathTextBox.Text = dialog.FileName;
            }
        }

        private void SaveButton_Click(object sender, RoutedEventArgs e)
        {
            BinaryFormatter formatter = new BinaryFormatter();
            SaveFileDialog dialog = new SaveFileDialog();

            if (dialog.ShowDialog() == true)
            {
                try
                {
                    using (FileStream fs = new FileStream(dialog.FileName, FileMode.OpenOrCreate))
                    {
                        formatter.Serialize(fs, ParallelDataProcessing.Network);

                        MessageBox.Show("Объект сериализован");
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show("Не удалось сохранить сеть по указанному пути!\n" + ex.Message);
                }
            }
        }

        private void LoadButton_Click(object sender, RoutedEventArgs e)
        {
            BinaryFormatter formatter = new BinaryFormatter();
            OpenFileDialog dialog = new OpenFileDialog();

            if (dialog.ShowDialog() == true)
            {
                using (FileStream fs = new FileStream(dialog.FileName, FileMode.OpenOrCreate))
                {
                    try
                    {
                        ConvolutionalNetwork network = (ConvolutionalNetwork)formatter.Deserialize(fs);
                        CnnTeacher teacher = new CnnTeacher(network);
                        ParallelDataProcessing = new ParallelDataProcessing(network, teacher);

                        MessageBox.Show("Объект десериализован");
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show("Не удалось загрузить объект сети из указанного файла!\n" + ex.Message);
                    }
                }
            }
        }

        private void LayersGrid_KeyUp(object sender, KeyEventArgs e)
        {
            if (e.Key != Key.Enter) return;

            int index = LayersGrid.SelectedIndex - 1;
            Layers.Insert(index, new LayerDescription());
            LayersGrid.ItemsSource = null;
            LayersGrid.ItemsSource = Layers;
            
            LayersGrid.UpdateLayout();
            LayersGrid.SelectedIndex = index;

            Console.WriteLine("Added");
        }

        private void LayersGrid_MouseDoubleClick(object sender, MouseButtonEventArgs e)
        {
            Console.WriteLine(LayersGrid.SelectedIndex);
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            Tensor3D tensor = new Tensor3D(56, 56, 313);
            SoftmaxLayer layer = new SoftmaxLayer();
            Stopwatch sw = new Stopwatch();
            long time = 0;

            layer.CalcSizes(new Tensor3DSize(313, 56, 56));

            for (int i = 0; i < 100; i++)
            {
                sw.Start();
                layer.Transfer(tensor);
                sw.Stop();
                time += sw.ElapsedTicks;
                sw.Reset();
            }
            

            Console.WriteLine($"Time: {time  / 100}");
        }
    }
}

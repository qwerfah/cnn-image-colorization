using CNN.DataProcessing;
using Microsoft.Win32;
using System.Drawing;
using System.Windows;

namespace CNN
{
    /// <summary>
    /// Логика взаимодействия для ImagesWindow.xaml
    /// </summary>
    public partial class ImagesWindow : Window
    {
        private readonly Bitmap _result = null;
        private readonly Bitmap _original = null;

        public ImagesWindow(Bitmap inputImage, Bitmap outputImage, Bitmap processedImage, long time)
        {
            InitializeComponent();

            InputImage.Source = IOProcessing.ConvertBitmapToSource(inputImage);
            OutputImage.Source = IOProcessing.ConvertBitmapToSource(outputImage);
            ProcesedImage.Source = IOProcessing.ConvertBitmapToSource(processedImage);
            TimeTextBox.Text = time.ToString();

            _original = outputImage;
            _result = processedImage;
        }

        private void SaveButton_Click(object sender, RoutedEventArgs e)
        {
            SaveFileDialog dialog = new SaveFileDialog();

            if (dialog.ShowDialog() == true)
            {
                _result.Save(dialog.FileName);
                _original.Save(dialog.FileName + "_orig");
            }
        }
    }
}

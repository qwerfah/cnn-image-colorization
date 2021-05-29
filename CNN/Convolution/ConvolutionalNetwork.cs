using CNN.AuxiliaryStructures;
using CNN.Convolution.Layers;
using CNN.Convolution.Layers.Functions;
using CNN.DataProcessing;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace CNN.Convolution
{
    [Serializable]
    public class ConvolutionalNetwork
    {
        private Tensor3DSize _inputSize;

        public Tensor3DSize InputSize 
        { 
            get
            {
                return _inputSize;
            }
            set
            {
                _inputSize = value;
                CalcLayerSizes();
            }
        }

        private List<Layer> _layers = null;

        public ConvolutionalNetwork(Tensor3DSize inputSize)
        {
            InputSize = inputSize;
            _layers = new List<Layer>();
        }

        public void AddLayer(Layer layer)
        {
            _layers.Add(layer);
            CalcLayerSizes();
        }

        public void AddLayers(IEnumerable<Layer> layers)
        {
            _layers.AddRange(layers);
            CalcLayerSizes();
        }

        public void ClearLayers()
        {
            _layers.Clear();
        }

        private void CalcLayerSizes()
        {
            if (_layers == null) return;
            Tensor3DSize size = InputSize;

            for (int i = 0; i < _layers.Count(); i++)
            {
                size = _layers[i].CalcSizes(size);
            }
        }

        // Обработка одного изображения с сохранием промежуточных значений
        public Tensor3D[] Transfer(Tensor3D image)
        {
            Tensor3D[] results = new Tensor3D[_layers.Count() + 1];
            results[0] = image;

            for (int i = 0; i < _layers.Count(); i++)
            {
                results[i + 1] = _layers[i].Transfer(results[i]);
            }

            return results;
        }

        // Обработка одного изображения
        public Tensor3D TransferSingle(Tensor3D image)
        {
            Tensor3D intermediate = image;

            for (int i = 0; i < _layers.Count(); i++)
            {
                intermediate = _layers[i].Transfer(intermediate);
            }

            return intermediate;
        }

        // Обработка одного изображения
        public void Transfer(object sender, DoWorkEventArgs e)
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();
            Tensor3D intermediate = (e.Argument as Tensor3D?).Value;
            float step = (float)Math.Round(100.0f / (float)_layers.Count());

            for (int i = 0; i < _layers.Count(); i++)
            {
                intermediate = _layers[i].Transfer(intermediate);
                (sender as BackgroundWorker).ReportProgress((int)step);
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }

            sw.Stop();

            e.Result = new CnnResult(intermediate, sw.ElapsedMilliseconds);
        }

        // Параллельная обработка массива изображений (среднее время обработки меньше)
        public Tensor3D[] Transfer(Tensor3D[] images)
        {
            Tensor3D[] result = new Tensor3D[images.Length];

            Parallel.For(0, images.Length, i =>
            {
                result[i] = TransferSingle(images[i]);
            });

            return result;
        }

        // Обратное распространение ошибки для пакета
        public void BackPropagation(Image[] images)
        {
            Parallel.For(0, images.Length, i =>
            {
                BackPropagation(images[i].Input, images[i].Output);
            });
        }

        // Обратное распространение ошибки для одного изображения
        public void BackPropagation(Tensor3D image, Tensor3D correct)
        {
            Tensor3D[] results = Transfer(image);
            Tensor3D deltas = _layers.Last().CalcOutputDeltas(results.Last(), correct);

            for (int i = _layers.Count() - 1; i >= 0; i--)
            {
                deltas = _layers[i].BackPropagation(deltas, results[i], results[i + 1]);
            }
        }

        public void UpdateWeights(float rate)
        {
            foreach (Layer layer in _layers)
            {
                if (layer is LearnableLayer)
                {
                    (layer as LearnableLayer).UpdateWeightsSGD(rate);
                }
                else if (layer is BatchNormLayer)
                {
                    (layer as BatchNormLayer).UpdateWeights(rate);
                }
            }
        }

        public void UpdateWeights(float rate, float pulse, float momentum)
        {
            foreach (Layer layer in _layers)
            {
                if (layer is LearnableLayer)
                {
                    (layer as LearnableLayer).UpdateWeightsAdam(rate, pulse, momentum);
                }
            }
        }

        public void LoadWeights()
        {
            Stream stream = new FileStream(
                Directory.GetCurrentDirectory() + @"\..\..\..\resources\ColorfulImageColorization.model", FileMode.Open);
            BinaryReader reader = new BinaryReader(stream);

            foreach (Layer layer in _layers)
            {
                if (layer is LearnableLayer)
                {
                    (layer as LearnableLayer).LoadWeights(reader);
                }
                else if (layer is BatchNormLayer)
                {
                    (layer as BatchNormLayer).LoadWeights(reader);
                }
            }

            reader.Close();
        }
    }
}

using CNN.AuxiliaryStructures;
using CNN.Convolution.Layers.Functions;
using System;
using System.Threading.Tasks;

namespace CNN.Convolution.Layers
{
    /// <summary>
    /// Слой функции активации.
    /// </summary>
    [Serializable]
    public class ActivationLayer : Layer
    {
        private ActivationFunction _activationFunction;
        private Derivative _derivative;

        public ActivationLayer(ActivationType activation)
        {
            _activationFunction = ActivationFunctions.Functions[(int)activation];
            _derivative = ActivationFunctions.Derivatives[(int)activation];
        }

        public override Tensor3D Transfer(Tensor3D data)
        {
            Tensor3D maps = new Tensor3D(_outputSize.Width, _outputSize.Height, _outputSize.Depth);

            Parallel.For(0, _inputSize.Depth, d =>
            {
                for (int y = 0; y < _inputSize.Height; y++)
                {
                    for (int x = 0; x < _inputSize.Width; x++)
                    {
                        maps.Set(x, y, d, _activationFunction(data.Get(x, y, d)));
                    }
                }
            });

            return maps;
        }

        public override Tensor3D BackPropagation(Tensor3D nextLayerDeltas, Tensor3D input, Tensor3D output)
        {
            Tensor3D deltas = new Tensor3D(_outputSize.Width, _outputSize.Height, _outputSize.Depth);

            Parallel.For(0, _inputSize.Depth, d =>
            {
                for (int y = 0; y < _outputSize.Height; y++)
                {
                    for (int x = 0; x < _outputSize.Width; x++)
                    {
                        deltas.Set(x, y, d, nextLayerDeltas.Get(x, y, d) * _derivative(output.Get(x, y, d)));
                    }
                }
            });

            return deltas;
        }

        public override Tensor3D CalcOutputDeltas(Tensor3D result, Tensor3D correct)
        {
            Tensor3D deltas = new Tensor3D(_outputSize.Width, _outputSize.Height, _outputSize.Depth);

            Parallel.For(0, _outputSize.Depth, d =>
            {
                for (int y = 0; y < _outputSize.Height; y++)
                {
                    for (int x = 0; x < _outputSize.Width; x++)
                    {
                        deltas.Set(x, y, d, CrossEntropyError(correct.Get(x, y, d), result.Get(x, y, d)));
                    }
                }
            });

            return deltas;
        }

        public static float AverageSquareError(float t, float y)
        {
            return (y - t);
        }

        public static float CrossEntropyError(float t, float y)
        {
            return -(t / y);
        }

        public override Tensor3DSize CalcSizes(Tensor3DSize inputSize)
        {
            _inputSize = inputSize;
            _outputSize = inputSize;

            return inputSize;
        }
    }
}

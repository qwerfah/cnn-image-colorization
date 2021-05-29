using CNN.AuxiliaryStructures;
using CNN.Convolution;
using CNN.Convolution.Layers;
using CNN.Convolution.Layers.Functions;
using System;
using System.Collections.Generic;

namespace CNN.DataProcessing
{
    public enum CnnArchitecture
    {
        VGG,
        VGGModified,
        Rec,
        None
    }

    public static class DescriptionToLayerConverter
    {
        public static Layer DescriptionToLayer(LayerDescription description)
        {
            switch (description.LayerType)
            {
                case LayerType.Convolution:
                    return new ConvolutionalLayer(
                        new Tensor3DSize(description.FilterDepth,
                        description.KernelHeight, description.KernelWidth),
                        description.FilterCount, description.Strides, description.Dilation);

                case LayerType.Subsampling:
                    return new SubsamplingLayer(description.PoolingType,
                        description.KernelWidth, description.KernelHeight);

                case LayerType.Upsampling:
                    return new UpsamplingLayer(
                        UpsamplingType.NearestNeighborUpsampling,
                        description.KernelWidth, description.KernelHeight);

                case LayerType.Deconvolution:
                    return new DeconvolutionalLayer(
                        new Tensor3DSize(description.FilterDepth,
                        description.KernelHeight, description.KernelWidth),
                        description.FilterCount, description.Strides);

                case LayerType.Activation:
                    return new ActivationLayer(description.ActivationType);

                case LayerType.BatchNorm:
                    return new BatchNormLayer();

                case LayerType.Softmax:
                    return new SoftmaxLayer();

                default: return null;
            }
        }

        public static List<Layer> DescriptionToLayer(IEnumerable<LayerDescription> descriptions)
        {
            List<Layer> layers = new List<Layer>();

            foreach (LayerDescription description in descriptions)
            {
                Layer layer = DescriptionToLayer(description);
                if (layer == null)
                {
                    throw new ArgumentException("Неверное описание слоя, невозможно сгенерировать структуру сети!");
                }
                layers.Add(layer);
            }

            return layers;
        }

        public static Pair<List<LayerDescription>, List<Layer>> GenerateArchitecture(CnnArchitecture architecture)
        {
            switch (architecture)
            {
                case CnnArchitecture.VGG:
                    return new Pair<List<LayerDescription>, List<Layer>>(VggArchitecture(), 
                        DescriptionToLayer(VggArchitecture()));
                case CnnArchitecture.VGGModified:
                    return new Pair<List<LayerDescription>, List<Layer>>(VggModifiedArchitecture(), 
                        DescriptionToLayer(VggModifiedArchitecture()));
                case CnnArchitecture.Rec:
                    return new Pair<List<LayerDescription>, List<Layer>>(RecArchitecture(),
                        DescriptionToLayer(RecArchitecture()));
                default:
                    return new Pair<List<LayerDescription>, List<Layer>>();
            }
        }

        public static List<LayerDescription> VggArchitecture()
        {
            return new List<LayerDescription>
            {
                // Conv 1
                ConvolutionalLayer(1, 64, 3), // 3x3
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(64, 64, 3, 2), // 5x5
                ActivationLayer(ActivationType.reLU),
                BatchNormLayer(),
                // Conv 2
                ConvolutionalLayer(64, 128, 3), // 9x9
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(128, 128, 3, 2), // 11x11
                ActivationLayer(ActivationType.reLU),
                BatchNormLayer(),
                // Conv 3
                ConvolutionalLayer(128, 256, 3), // 15x15
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(256, 256, 3), // 17x17
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(256, 256, 3, 2), // 19x19
                ActivationLayer(ActivationType.reLU),
                BatchNormLayer(),
                // Conv 4
                ConvolutionalLayer(256, 512, 3),  // 23x23
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(512, 512, 3), // 25x25
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(512, 512, 3), // 27x27
                ActivationLayer(ActivationType.reLU),
                BatchNormLayer(),
                // Conv 5
                ConvolutionalLayer(512, 512, 3, 1, 2), // 31x31
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(512, 512, 3, 1, 2), // 35x35
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(512, 512, 3, 1, 2), // 39x39
                ActivationLayer(ActivationType.reLU),
                BatchNormLayer(),
                // Conv 6
                ConvolutionalLayer(512, 512, 3, 1, 2), // 43x43
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(512, 512, 3, 1, 2), // 47x47
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(512, 512, 3, 1, 2), // 51x51
                ActivationLayer(ActivationType.reLU),
                BatchNormLayer(),
                // Conv 7
                ConvolutionalLayer(512, 512, 3), // 53x53
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(512, 512, 3), // 55x55
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(512, 512, 3), // 57x57
                ActivationLayer(ActivationType.reLU),
                BatchNormLayer(),
                // Conv 8
                DeconvolutionalLayer(256, 512, 4, 2),
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(256, 256, 3),
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(256, 256, 3),
                ActivationLayer(ActivationType.reLU),
                // Softmax
                ConvolutionalLayer(256, 313, 1),
                ActivationLayer(ActivationType.Linear),
                SoftmaxLayer(),
                ConvolutionalLayer(313, 2, 1),
            };
        }

        public static List<LayerDescription> RecArchitecture()
        {
            return new List<LayerDescription>
            {
                // Conv 1
                ConvolutionalLayer(1, 64, 3), // 3x3
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(64, 64, 3, 2), // 5x5
                ActivationLayer(ActivationType.reLU),
                BatchNormLayer(),
                // Conv 2
                ConvolutionalLayer(64, 128, 3), // 9x9
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(128, 128, 3, 2), // 11x11
                ActivationLayer(ActivationType.reLU),
                BatchNormLayer(),
                // Conv 3
                ConvolutionalLayer(128, 256, 3), // 15x15
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(256, 256, 3), // 17x17
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(256, 256, 3, 2), // 19x19
                ActivationLayer(ActivationType.reLU),
                BatchNormLayer(),
                // Conv 4
                ConvolutionalLayer(256, 512, 3),  // 23x23
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(512, 512, 3), // 25x25
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(512, 512, 3), // 27x27
                ActivationLayer(ActivationType.reLU),
                BatchNormLayer(),
                // Conv 5
                ConvolutionalLayer(512, 512, 3, 1, 2), // 31x31
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(512, 512, 3, 1, 2), // 35x35
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(512, 512, 3, 1, 2), // 39x39
                ActivationLayer(ActivationType.reLU),
                BatchNormLayer(),
                // Conv 6
                ConvolutionalLayer(512, 512, 3, 1, 2), // 43x43
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(512, 512, 3, 1, 2), // 47x47
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(512, 512, 3, 1, 2), // 51x51
                ActivationLayer(ActivationType.reLU),
                BatchNormLayer(),
                // Conv 7
                ConvolutionalLayer(512, 512, 3), // 53x53
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(512, 512, 3), // 55x55
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(512, 512, 3), // 57x57
                ActivationLayer(ActivationType.reLU),
                BatchNormLayer(),
                // Conv 8
                DeconvolutionalLayer(256, 512, 4, 2),
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(256, 256, 3),
                ActivationLayer(ActivationType.reLU),

                DeconvolutionalLayer(256, 256, 4, 2),
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(256, 256, 3),
                ActivationLayer(ActivationType.reLU),

                DeconvolutionalLayer(256, 256, 4, 2),
                ActivationLayer(ActivationType.reLU),
                // Softmax
                ConvolutionalLayer(256, 313, 1),
                ActivationLayer(ActivationType.Linear),
                SoftmaxLayer(),
                ConvolutionalLayer(313, 2, 1),
            };
        }

        public static List<LayerDescription> VggModifiedArchitecture()
        {
            return new List<LayerDescription>
            {
                // Conv 1
                ConvolutionalLayer(1, 64, 3),
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(64, 64, 3, 2),
                ActivationLayer(ActivationType.reLU),
                // Conv 2
                ConvolutionalLayer(64, 128, 3),
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(128, 128, 3, 2),
                ActivationLayer(ActivationType.reLU),
                // Conv 3
                ConvolutionalLayer(128, 256, 3),
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(256, 256, 3),
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(256, 256, 3, 2),
                ActivationLayer(ActivationType.reLU),
                // Conv 4
                ConvolutionalLayer(256, 512, 3),
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(512, 512, 3),
                ActivationLayer(ActivationType.reLU),
                ConvolutionalLayer(512, 512, 3),
                ActivationLayer(ActivationType.reLU),
                // Conv 8
                DeconvolutionalLayer(256, 512, 4, 2),
                ActivationLayer(ActivationType.reLU),
                DeconvolutionalLayer(256, 256, 4, 2),
                ActivationLayer(ActivationType.reLU),
                // Softmax
                ConvolutionalLayer(256, 256, 1),
                ActivationLayer(ActivationType.Linear),
                SoftmaxLayer(),
                ConvolutionalLayer(256, 2, 1),
            };
        }

        public static LayerDescription 
            ConvolutionalLayer(int oldDepth, int newDepth, int kernelSize = 3, int strides = 1, int dilation = 1)
        {
            return new LayerDescription
            {
                LayerType = LayerType.Convolution,
                FilterDepth = oldDepth,
                FilterCount = newDepth,
                KernelHeight = kernelSize,
                KernelWidth = kernelSize,
                Strides = strides,
                Dilation = dilation
            };
        }

        public static LayerDescription 
            DeconvolutionalLayer(int oldDepth, int newDepth, int kernelSize = 3, int strides = 1)
        {
            return new LayerDescription
            {
                LayerType = LayerType.Deconvolution,
                FilterDepth = oldDepth,
                FilterCount = newDepth,
                KernelHeight = kernelSize,
                KernelWidth = kernelSize,
                Strides = strides,
            };
        }

        public static LayerDescription ActivationLayer(ActivationType activation)
        {
            return new LayerDescription
            {
                LayerType = LayerType.Activation,
                ActivationType = activation
            };
        }

        public static LayerDescription BatchNormLayer()
        {
            return new LayerDescription
            {
                LayerType = LayerType.BatchNorm,
            };
        }

        public static LayerDescription SoftmaxLayer()
        {
            return new LayerDescription
            {
                LayerType = LayerType.Softmax,
            };
        }

        public static LayerDescription UpsamplingLayer(int scaleX, int scaleY)
        {
            return new LayerDescription
            {
                LayerType = LayerType.Upsampling,
                KernelHeight = scaleY,
                KernelWidth = scaleX,
            };
        }

        public static LayerDescription SubsamplingLayer(int scaleX, int scaleY)
        {
            return new LayerDescription
            {
                LayerType = LayerType.Deconvolution,
                KernelHeight = scaleY,
                KernelWidth = scaleX,
            };
        }
    }
}

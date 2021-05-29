using System;
using System.Runtime.CompilerServices;

namespace CNN.Convolution.Layers.Functions
{
    /// <summary>
    /// Тип функции активаци сверточной нейронной сети.
    /// </summary>
    public enum ActivationType
    {
        reLU,
        LeakyReLU,
        ELU,
        Sigmoid,
        Tanh,
        Linear,
        None
    }

    /// <summary>
    /// Делегат функции активации сверточной нейронной сети.
    /// </summary>
    /// <param name="s">Взвешенная сумма входных сигналов нейрона.</param>
    /// <returns></returns>
    public delegate float ActivationFunction(float s);
    /// <summary>
    /// Делегат производной функции активации сверточной нейронной сети.
    /// </summary>
    /// <param name="s">Значение функции активации/значение на входе функции.</param>
    /// <returns></returns>
    public delegate float Derivative(float s);

    /// <summary>
    /// Предоставляет функции активации нейронов сверточной нейронной сети и их производные.
    /// </summary>
    public static class ActivationFunctions
    {
        public static ActivationFunction[] Functions =
        {
            ReLU,
            LeakyReLU,
            ELU,
            Sigmoid,
            Tanh,
            Linear,
            None
        };

        public static Derivative[] Derivatives =
        {
            ReLUDerivative,
            LeakyReLUDerivative,
            ELUDerivative,
            SigmoidDerivative,
            TanhDerivative,
            LinearDerivative,
            NoneDerivative
        };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ReLU(float s)
        {
            if (float.IsInfinity(s) || float.IsNaN(s))
            {
                throw new ArithmeticException("Ошибка рапространения: получено NaN или Infinity!");
            }
            return (s >= 0f) ? s : 0.0f;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ReLUDerivative(float s)
        {
            if (float.IsInfinity(s) || float.IsNaN(s))
            {
                throw new ArithmeticException("Ошибка обучения: получено NaN или Infinity!");
            }
            return (s >= 0f) ? 1.0f : 0.0f;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float LeakyReLU(float s)
        {
            if (float.IsInfinity(s) || float.IsNaN(s))
            {
                throw new ArithmeticException("Ошибка рапространения: получено NaN или Infinity!");
            }
            return (s >= 0f) ? s : (0.01f * s);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float LeakyReLUDerivative(float s)
        {
            if (float.IsInfinity(s) || float.IsNaN(s))
            {
                throw new ArithmeticException("Ошибка обучения: получено NaN или Infinity!");
            }
            return (s >= 0f) ? 1.0f : 0.01f;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ELU(float s)
        {
            if (float.IsInfinity(s) || float.IsNaN(s))
            {
                throw new ArithmeticException("Ошибка рапространения: получено NaN или Infinity!");
            }
            return (s >= 0f) ? s : 0.1f * ((float)Math.Exp(s) - 1f);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ELUDerivative(float s)
        {
            if (float.IsInfinity(s) || float.IsNaN(s))
            {
                throw new ArithmeticException("Ошибка рапространения: получено NaN или Infinity!");
            }
            return (s >= 0f) ? 1.0f : (s + 0.1f);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sigmoid(float s)
        {
            if (float.IsInfinity(s) || float.IsNaN(s))
            {
                throw new ArithmeticException("Ошибка рапространения: получено NaN или Infinity!");
            }
            return 1.0f / (1.0f + (float)Math.Exp(-s));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SigmoidDerivative(float s)
        {
            if (float.IsInfinity(s) || float.IsNaN(s))
            {
                throw new ArithmeticException("Ошибка обучения: получено NaN или Infinity!");
            }
            return s * (1.0f - s);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Tanh(float s)
        {
            if (float.IsInfinity(s) || float.IsNaN(s))
            {
                throw new ArithmeticException("Ошибка рапространения: получено NaN или Infinity!");
            }
            return (1.0f - (2.0f / ((float)Math.Exp(2.0f * s) + 1.0f)));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float TanhDerivative(float s)
        {
            if (float.IsInfinity(s) || float.IsNaN(s))
            {
                throw new ArithmeticException("Ошибка рапространения: получено NaN или Infinity!");
            }
            return (1.0f - (s * s));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Linear(float s)
        {
            if (float.IsInfinity(s) || float.IsNaN(s))
            {
                throw new ArithmeticException("Ошибка рапространения: получено NaN или Infinity!");
            }
            return 6.0f * s;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float LinearDerivative(float s)
        {
            if (float.IsInfinity(s) || float.IsNaN(s))
            {
                throw new ArithmeticException("Ошибка рапространения: получено NaN или Infinity!");
            }
            return 6.0f;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float None(float s)
        {
            return s;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float NoneDerivative(float s)
        {
            return 1.0f;
        }
    }
}

using System;

namespace CNN.DataProcessing
{
    public static class Settings
    {
        public static int ImageWidth { get; set; } = 224;
        public static int ImageHeight { get; set; } = 224;
        public static Random Random { get; set; } = new Random(DateTime.Now.Millisecond);
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FaceBot
{
    // Used to store configurable values that are used across the entire project for various purposes. 
    public class Config
    {
        public static readonly string FaceHaarCascadeFilePath = "/haarcascade_frontalface_default.xml";
        public static readonly string DatabaseFilePath = "./Database/facebot.db";
        public static readonly string TrainingDataFilePath = "./Database/trainingdata.yaml";
        public static readonly double FacesDetectorInterval = 1000;
        public static readonly double FacesRecognizerInterval = 3000;
        public static readonly string UnrecognizedUserName = "UNRECOGNIZED";
        public static readonly string DefaultBrowserProcessName = "chrome";
    }
}

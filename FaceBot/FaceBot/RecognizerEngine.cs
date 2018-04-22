using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Face;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Emgu.CV.Face.FaceRecognizer;

namespace FaceBot
{
    class RecognizerEngine
    {
        // LBPH Face Recognizer.
        private FaceRecognizer _faceRecognizer;
        private DataStoreAccess _dataStoreAccess;
        private String _recognizerFilePath;

        public RecognizerEngine(String databasePath, String recognizerFilePath)
        {
            _recognizerFilePath = recognizerFilePath;
            _dataStoreAccess = new DataStoreAccess(databasePath);
            _faceRecognizer = new LBPHFaceRecognizer();
        }

        // Train the recognizer by fetching ALL user faces along with their associated user IDs from the database.
        public bool TrainRecognizer()
        {
            var allFaces = _dataStoreAccess.GetFaces();
            if (allFaces != null)
            {
                var faceImages = new Image<Gray, byte>[allFaces.Count];
                var faceIds = new int[allFaces.Count];
                for (int i = 0; i < allFaces.Count; i++)
                {
                    Stream stream = new MemoryStream();
                    stream.Write(allFaces[i].Image, 0, allFaces[i].Image.Length);
                    var faceImage = new Image<Gray, byte>(new Bitmap(stream));
                    faceImages[i] = faceImage.Resize(100, 100, Inter.Cubic);
                    faceIds[i] = allFaces[i].Id;
                }
                _faceRecognizer.Train(faceImages, faceIds);
                _faceRecognizer.Save(_recognizerFilePath);
            }
            return true;

        }

        // Attempt to recognize the user using the parameterized image & the images in the database. 
        public User RecognizeUser(Image<Gray, byte> userImage)
        {
            _faceRecognizer.Load(_recognizerFilePath);
            var faceData = _faceRecognizer.Predict(userImage.Resize(100, 100, Inter.Cubic));

            // Threshold value set to 87 after thorough testing & evaluation with a database of 50+ images from the internet. 
            if (faceData.Distance < 87)
            {
                var user = _dataStoreAccess.GetUserByFaceId(faceData.Label);
                user.Distance = faceData.Distance;
                return user;
            }
            else
                return new User { UserName = Config.UnrecognizedUserName, Distance = faceData.Distance }; 
        }
    }
}

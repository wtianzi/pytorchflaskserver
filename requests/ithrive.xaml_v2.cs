using Microsoft.ML;
using OnnxObjectDetection;
using OpenCvSharp;

using System;
using System.Collections.Generic;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging; 
 
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks; 
using Rectangle = System.Windows.Shapes.Rectangle;
 
using Microsoft.Win32;
using System.Diagnostics;
using Microsoft.ML.Data;

using Microsoft.ML.Transforms.Image;
using System.Drawing;

namespace OnnxObjectDetectionApp
{
    /// <summary>
    /// Interaction logic for ithrive.xaml
    /// </summary>
    public partial class ithrive : System.Windows.Window
    {
        //private MLContext context;
        

        private OnnxOutputParser outputParser;
        private static readonly string modelsDirectory = Path.Combine(Environment.CurrentDirectory, @"ML\OnnxModels");
        private string model_name = Path.Combine(modelsDirectory, "model_fine.onnx");
        private PredictionEngine<ImageInputData, eyegazePrediction> eyegazePredictionEngine;

        public ithrive()
        {
            InitializeComponent();
            //LoadModel();
        }
        
        private void button_Click_1(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            if (openFileDialog.ShowDialog() == true)
            {
                var bitmap = new BitmapImage(new Uri(openFileDialog.FileName));
                image.Source = bitmap;

                var frame = new ImageInputData { Image = BitmapImage2Bitmap(bitmap) };
                var filteredBoxes = DetectObjectsUsingModel(frame);

                DrawOverlays(filteredBoxes, image.ActualHeight, image.ActualWidth);
            }
                
        }
        private Bitmap BitmapImage2Bitmap(BitmapImage bitmapImage)
        {
            // BitmapImage bitmapImage = new BitmapImage(new Uri("../Images/test.png", UriKind.Relative));

            using (MemoryStream outStream = new MemoryStream())
            {
                BitmapEncoder enc = new BmpBitmapEncoder();
                enc.Frames.Add(BitmapFrame.Create(bitmapImage));
                enc.Save(outStream);
                System.Drawing.Bitmap bitmap = new System.Drawing.Bitmap(outStream);

                return new Bitmap(bitmap);
            }
        } 
        
        public List<BoundingBox> DetectObjectsUsingModel_eye(ImageInputData imageInputData)
        {
            var labels = eyegazePredictionEngine?.Predict(imageInputData).PredictedLabels;
            var boundingBoxes = outputParser.ParseOutputs(labels);
            var filteredBoxes = outputParser.FilterBoundingBoxes(boundingBoxes, 5, 0.5f);
            return filteredBoxes;
        }
        
        private void button1_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            if (openFileDialog.ShowDialog() == true)
            {
                var bitmap = new BitmapImage(new Uri(openFileDialog.FileName));
                image.Source = bitmap;
                var frame = new ImageInputData { Image = BitmapImage2Bitmap(bitmap) };
                
                var mlContext = new MLContext();

                var eyegazeModel = new EyegazeModel(model_name);

                var dataView = mlContext.Data.LoadFromEnumerable(new List<eyegazeInputData>());

                var pipeline = mlContext.Transforms.ApplyOnnxModel(modelFile: eyegazeModel.ModelPath, outputColumnName: eyegazeModel.ModelOutput, inputColumnName: eyegazeModel.ModelInput);
                var mlNetModel = pipeline.Fit(dataView);

                mlContext.Model.CreatePredictionEngine<ImageInputData[], List<eyegazeInputData>>(mlNetModel);

                var results = eyegazePredictionEngine?.Predict(imageInputData).PredictedLabels;
            }

        }
    }
    public class EyegazeModel : IOnnxModel
    {
        public string ModelPath { get; private set; }

        public string ModelInput { get; } = "input";
        public string ModelOutput { get; } = "output";
        
        public EyegazeModel(string modelPath)
        {
            ModelPath = modelPath;
        }
    }

    public class eyegazePrediction : IOnnxObjectPrediction
    {
        [ColumnName("output")]
        public float[] PredictedLabels { get; set; }
    }

    public class eyegazeInputData
    {
        [ColumnName("input")]
        [VectorType(1,3,16,64,64)]
        public Single[] input { get; set; }
    }
    
}

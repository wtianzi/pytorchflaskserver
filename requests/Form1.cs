using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using Accord.Video;
using Accord.Video.DirectShow;
using Accord.Video.FFMPEG;
using Accord.Video.VFW;
using Tobii.Research;

using System.IO;
using System.Diagnostics;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

using Flir.Atlas.Image;
using Flir.Atlas.Live.Device;
using Flir.Atlas.Live.Discovery;

using System.Timers;
// This is the code for your desktop app.
// Press Ctrl+F5 (or go to Debug > Start Without Debugging) to run your app.

//use remote model
using System.Net;
using System.Collections.Specialized;
using System.Runtime.InteropServices;
using Emgu.CV.Util;
using Emgu.Util;
using Emgu.Util.TypeEnum;
namespace TobiiTesting1
{
    
    public partial class Form1 : Form
    {
        private FilterInfoCollection videoDevices;
        private VideoCaptureDevice videoSource = null;        

        //private FilterInfoCollection VideoCaptureDevices;
        //private VideoCaptureDevice FinalVideo = null;
        //private VideoCaptureDeviceForm captureDevice;
        private Bitmap videoimg;
        //private AVIWriter AVIwriter = new AVIWriter();
        //private VideoFileWriter FileWriter = new VideoFileWriter();
        private SaveFileDialog saveAvi;
        public static bool startsession;
        public static bool eyetrackingrecordenabled = false;

        private string trialsavingpath = "";

        static string gazedatasavingpath = "";

        private string m_savingfolder = "";
        //static bool b_recording = false;

        static float m_eyegazex, m_eyegazey;
        static string m_eyegazestr = "No Eye Tracker";

        // true means the trial is in use, false means the trial is not in use
        // start recording: b_trial_locked=true, trial index;
        // end trial b_trial_locked=false,
        // start a new trial: b_trial_locked=true
        private bool b_trial_locked = true;
        //FormCamera cameraForm = new FormCamera();
        List<FormCameras> m_cameras = new List<FormCameras>();
        List<MainWindow> m_thermalcams = new List<MainWindow>();

        //Form_Empatica empaticaForm;

        public bool m_empaticarunning;
        //private CascadeClassifier _cascadeClassifier;

        private DateTime m_trialstart = DateTime.Now;

        private string str_trial = "";

        private string[] task_list = { "Task A : Peg Transfer", "Task B : Precision Cutting", "Task C : Ligating Loop", "Task D : Suture with Extracorporeal Knot", "Task E : Suture with Intracorporeal Knot" };
        
        private string[] task_list_log = { "A", "B", "C", "D", "E"};

        private string[,] task_performance = new string[5, 4] {
                {"#of object dropped outside field of view","N/A","N/A","N/A"},
                {"All cuts within the lines? YES: 0, NO: 1","N/A","N/A","N/A"},
                {"Loop is secure? YES: 0, NO: 1","Loop is ___mm away from mark on appendage","N/A","N/A"},
                {"Knot is secure? YES: 0, NO: 1","Slit in drain is closed? YES: 0, NO: 1","Suture is ___mm away from dots","Drain was avulsed from foam block? YES: 0, NO: 1"},
                {"Knot is secure? YES: 0, NO: 1","Slit in drain is closed? YES: 0, NO: 1","Suture is ___mm away from dots","Drain was avulsed from foam block? YES: 0, NO: 1"}
            };

        private static AsynchronousClientD m_empatica_0 = new AsynchronousClientD();
        private static AsynchronousClientD m_empatica_1 = new AsynchronousClientD();

        private static System.Timers.Timer aTimer;
        private static bool m_starttimer;

        private static float m_eyetrackerfrequency = 120;
        private WebClient m_wb = new WebClient();
        //private List<> m_imglist = new List<T>();
        private static SortedList<int, string> m_sortedlist = new SortedList<int, string>();
        private static string m_tempfolder = "tempimages/";
        private static int m_imagesavecount = 0;
        private static string url = "http://172.28.144.160:5000/predict";//"http://192.168.1.121:5000/predict";

        // draw rectangle on picturebox
        private static int m_x=10;
        private static int m_y=10;

        private static float xratio = 1;
        private static float yratio = 1;

        System.Drawing.Graphics formGraphics;
        public class ModelImage
        {
            public string filename;
            public int fileindex;
            public string results;
        }
        public Form1()
        {
            InitializeComponent();
            startsession = false;
            m_empaticarunning = false;
            
            //formGraphics = this.CreateGraphics(); 
            //xratio = pictureBox1.Width / 1024;
            //yratio = pictureBox1.Height / 768;

        }

        // get the devices name
        private void getCamList()
        {
            try
            {
                videoDevices = new FilterInfoCollection(FilterCategory.VideoInputDevice);
                //comboBox1.Items.Clear();
                if (videoDevices.Count == 0)
                    throw new ApplicationException();

                listView_CameraControl.Items.Clear();
                int t_index = 0;
                foreach (FilterInfo device in videoDevices)
                {
                    //comboBox1.Items.Add(device.Name);

                    ListViewItem item1 = new ListViewItem("", 0);
                    if (device.Name.Contains("eBUS"))
                    {
                        item1.Checked = false;

                    }
                    else
                    {
                        //item1.Checked = true;
                        item1.Checked = false;
                    }
                                       
                    item1.SubItems.Add(device.Name);
                    item1.SubItems.Add(t_index.ToString());
                    listView_CameraControl.Items.Add(item1);
                    t_index++;
                }
            }
            catch (ApplicationException)
            {
                //comboBox1.Items.Add("No capture device on your system");
            }
        }

        private void UpdateLabel2Text(String t_info)
        {
            label2.Text = t_info;
        }
       
        private Mat DrawInfoToImage(Mat img,string t_str="test")
        {
            CvInvoke.PutText(
               img,
               m_eyegazestr,
               new System.Drawing.Point(10, 80),
               FontFace.HersheyComplex,
               0.5,
               new Bgr(0, 255, 0).MCvScalar);
            return img;
        }
        

        //get total received frame at 1 second tick
        private void timer_main_Tick(object sender, EventArgs e)
        {
            //update timer
            TimeSpan t_min = DateTime.Now.Subtract(m_trialstart);
            label_time.Text = String.Format("{0:D2}:{1:D2}", t_min.Minutes, t_min.Seconds);
        }

        //prevent sudden close while device is running
        private void Form1_FormClosed(object sender, FormClosedEventArgs e)
        {
            aTimer.Enabled = false;
            
            foreach (FormCameras item in m_cameras)
            {
                item.CloseVideoSource();
            }
            foreach (MainWindow item in m_thermalcams)
            {
                item.CloseVideoSource();
            }
            //destroy the folder
            //formGraphics.Dispose();
        }
       
        private void rfsh_Click_1(object sender, EventArgs e)
        {
            getCamList();
            //comboBox1.SelectedIndex = 0; //make dafault to first cam
        }

        private void button2_Click(object sender, EventArgs e)
        {
            /*
            Browsing for eye trackers or selecting an eye tracker with known address.
            Establishing a connection with the eye tracker.
            Running a calibration procedure in which the eye tracker is calibrated to the user.
            Setting up a subscription to gaze data, and collecting and saving the data on the computer running the application.In some cases, the data is also shown live by the application.
            */
            
            var eyeTrackers = EyeTrackingOperations_FindAllEyeTrackers.Execute(this);
            while (eyeTrackers.Count < 1)
            {
                System.Threading.Thread.Sleep(2000);
                eyeTrackers = EyeTrackingOperations_FindAllEyeTrackers.Execute(this);
            }
            var eyeTracker = eyeTrackers[0];


            IEyeTracker_GazeOutputFrequencies.Execute(eyeTracker);
            label_pupil.Text = "eyetracker frequency: "+m_eyetrackerfrequency.ToString()+" Hz";

            CallEyeTrackerManager.Execute(eyeTracker);

            

            IEyeTracker_GazeDataReceived.Execute(eyeTracker,this);

        }

        internal static class IEyeTracker_GazeOutputFrequencies
        {
            internal static void Execute(IEyeTracker eyeTracker)
            {
                GazeOutputFrequencies(eyeTracker);
            }
            // <BeginExample>
            internal static void GazeOutputFrequencies(IEyeTracker eyeTracker)
            {
                Console.WriteLine("\nGaze output frequencies.");
                // Get and store current frequency so it can be restored.
                var initialGazeOutputFrequency = eyeTracker.GetGazeOutputFrequency();
                Console.WriteLine("Gaze output frequency is: {0} hertz.", initialGazeOutputFrequency);
                try
                {
                    // Get all gaze output frequencies.
                    var allGazeOutputFrequencies = eyeTracker.GetAllGazeOutputFrequencies();
                    foreach (var gazeOutputFrequency in allGazeOutputFrequencies)
                    {
                        if (gazeOutputFrequency < 110.0f)
                        {
                            initialGazeOutputFrequency = gazeOutputFrequency;
                            eyeTracker.SetGazeOutputFrequency(gazeOutputFrequency);
                            m_eyetrackerfrequency = gazeOutputFrequency;
                        }
                        //eyeTracker.SetGazeOutputFrequency(gazeOutputFrequency);
                        //Console.WriteLine("New gaze output frequency is: {0} hertz.", gazeOutputFrequency.ToString());
                    }
                }
                finally
                {
                    eyeTracker.SetGazeOutputFrequency(initialGazeOutputFrequency);
                    Console.WriteLine("Gaze output frequency reset to: {0} hertz.", eyeTracker.GetGazeOutputFrequency());
                }
            }
            // <EndExample>
        }

        internal static class EyeTrackingOperations_FindAllEyeTrackers
        {
            internal static EyeTrackerCollection Execute(Form1 formObject)
            {
                // <BeginExample>
                //Console.WriteLine("\nSearching for all eye trackers");
                EyeTrackerCollection eyeTrackers = EyeTrackingOperations.FindAllEyeTrackers();
                foreach (IEyeTracker eyeTracker in eyeTrackers)
                {
                    //Console.WriteLine("{0}, {1}, {2}, {3}, {4}, {5}", eyeTracker.Address, eyeTracker.DeviceName, eyeTracker.Model, eyeTracker.SerialNumber, eyeTracker.FirmwareVersion, eyeTracker.RuntimeVersion);
                    var t_str = String.Format("{0}, {1}, {2}, {3}, {4}, {5}", eyeTracker.Address, eyeTracker.DeviceName, eyeTracker.Model, eyeTracker.SerialNumber, eyeTracker.FirmwareVersion, eyeTracker.RuntimeVersion);
                    formObject.UpdateLabel2Text(t_str);
                }
                // <EndExample>
                return eyeTrackers;
            }
        }
        class IEyeTracker_GazeDataReceived
        {
            
            public static void Execute(IEyeTracker eyeTracker,Form1 formObject)
            {
                if (eyeTracker != null)
                {
                    GazeData(eyeTracker);
                }
            }
            // <BeginExample>
            private static void GazeData(IEyeTracker eyeTracker)
            {
                // Start listening to gaze data.

                eyeTracker.GazeDataReceived += EyeTracker_GazeDataReceived;
                // Wait for some data to be received.
                System.Threading.Thread.Sleep(2000);//2000
                // Stop listening to gaze data.
                //eyeTracker.GazeDataReceived -= EyeTracker_GazeDataReceived;
            }
            private static void EyeTracker_GazeDataReceived(object sender, GazeDataEventArgs e)
            {
                var local_timestamp = DateTimeOffset.Now.ToString("MM/dd/yyyy hh:mm:ss.fff").ToString();
                var UnixTimestamp = new DateTimeOffset(DateTime.UtcNow).ToUnixTimeMilliseconds().ToString();

                var t_str = String.Format("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12}\r\n",
                    e.LeftEye.GazeOrigin.Validity,
                    e.LeftEye.GazeOrigin.PositionInUserCoordinates.X,
                    e.LeftEye.GazeOrigin.PositionInUserCoordinates.Y,
                    e.LeftEye.GazeOrigin.PositionInUserCoordinates.Z,
                    e.LeftEye.GazePoint.PositionOnDisplayArea.X,
                    e.LeftEye.GazePoint.PositionOnDisplayArea.Y,
                    e.RightEye.GazePoint.PositionOnDisplayArea.X,
                    e.RightEye.GazePoint.PositionOnDisplayArea.Y,
                    e.LeftEye.Pupil.Validity,
                    e.LeftEye.Pupil.PupilDiameter,
                    e.RightEye.Pupil.PupilDiameter,
                    UnixTimestamp,
                local_timestamp);
                

               
                //project eye tracking data to form image
                // eye tracking data: screen center point (0,0)
                m_eyegazex = e.LeftEye.GazePoint.PositionOnDisplayArea.X;
                m_eyegazey = e.LeftEye.GazePoint.PositionOnDisplayArea.Y;

                m_eyegazestr = String.Format("{0},{1},{2}",
                   m_eyegazex,
                   m_eyegazey,
                   UnixTimestamp);


                //m_eyegazey = e.LeftEye.GazeOrigin.PositionInUserCoordinates.Y;
                /* 
                 (
                 "Gaze data with {0} left eye origin at point ({1}, {2}, {3}) in the user coordinate system. TimeStamp: {4}",
                 e.LeftEye.GazeOrigin.Validity,
                 e.LeftEye.GazeOrigin.PositionInUserCoordinates.X,
                 e.LeftEye.GazeOrigin.PositionInUserCoordinates.Y,
                 e.LeftEye.GazeOrigin.PositionInUserCoordinates.Z,
                 systemTimeStamp);
                 */

                //System.IO.File.WriteAllText(filename, t_str);
                if (eyetrackingrecordenabled)
                {
                    System.IO.File.AppendAllText(gazedatasavingpath, t_str);
                }
                
            }
            // <EndExample>
        }
        private void save_Click(object sender, EventArgs e)
        {
            saveAvi = new SaveFileDialog();
            //saveAvi.Filter = "Txt Files (*.txt)|*.txt";
            saveAvi.FileName = DateTimeOffset.Now.ToString("MM_dd_yy_hh_mm_ss");//saveAvi.FileName= DateTimeOffset.Now.ToString("MM_dd_yyyy hh_mm_ss");
            if (saveAvi.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                //create a new folder
                try
                {
                    m_savingfolder = saveAvi.FileName;

                    // Determine whether the directory exists.
                    if (Directory.Exists(m_savingfolder))
                    {
                        Console.WriteLine("That path exists already.");
                    }
                    else
                    {
                        DirectoryInfo di = Directory.CreateDirectory(m_savingfolder);
                        Console.WriteLine("The directory was created successfully at {0}.", Directory.GetCreationTime(m_savingfolder));
                    }
                    
                    GenerateRecordingFile();
                    
                    startsession = true;

                    button_endtask.Enabled = true;
                    textBox_participant.Enabled = true;
                    comboBox1.Enabled = true;
                    trialIndex.Enabled = true;
                    bt_trial.Enabled = true;

                    textBox_score_0.Enabled = true;
                    textBox_score_1.Enabled = true;
                    textBox_score_2.Enabled = true;
                    textBox_score_3.Enabled = true;
                    textBox_comment.Enabled = true;
                }
                catch 
                {
                    Console.WriteLine("The process failed: {0}", e.ToString());
                }
                finally
                {
                    save.Enabled = false;
                    System.Threading.Thread.Sleep(500);
                }

                //avisavingpath = saveAvi.FileName;
            }

        }
        

        
        private bool GenerateRecordingFile()
        {
            //for eyegazedata
            var local_timestamp = DateTimeOffset.Now.ToString("MM/dd/yyyy hh:mm:ss.fff").ToString();
            //string t_txtfilename = String.Format("_Tobii_{0}.txt", local_timestamp);

            var UnixTimestamp = new DateTimeOffset(DateTime.UtcNow).ToUnixTimeMilliseconds();

            
            //for trialsaving data
            trialsavingpath = m_savingfolder+ "\\Trials_" + UnixTimestamp.ToString()+".txt";

            if (!System.IO.File.Exists(trialsavingpath))
            {
                //create file
                using (var t_file = System.IO.File.Create(trialsavingpath));

                //trial format: participant,recordtype,index,status,score,unixtimestamp,localtimestamp,comment
                //participant,unixtimestamp,localtimestamp,recordtype,status,taskindex,trialindex,score,comment

                //string t_str = "participant,unixtimestamp,localtimestamp,recordtype,status,taskindex,trialindex,duration,score,comment\r\n";
                //\"participant\":,\"unixtimestamp\":,\"localtimestamp\":,\"recordtype\":,\"status\":,\"taskindex\":,\"trialindex\":,\"duration\":,\"score\":,\"comment\":
                //start session
                string t_str = "{\r\n";
                System.IO.File.AppendAllText(trialsavingpath, t_str);

                //t_str = String.Format("{0},{1},{2},SESSION,START,NA,NA,NA,NA,NA\r\n", textBox_participant.Text, UnixTimestamp, local_timestamp);

                t_str = String.Format("{{\"participant\":\"{0}\"," +
                    "\"unixtimestamp\":\"{1}\"," +
                    "\"localtimestamp\":\"{2}\"," +
                    "\"recordtype\":\"SESSION\"," +
                    "\"status\":\"START\"," +
                    "\"taskindex\":\"NA\"," +
                    "\"trialindex\":\"NA\"," +
                    "\"duration\":\"NA\"," +
                    "\"score\":\"NA\"," +
                    "\"comment\":\"NA\"}},\r\n", 
                    textBox_participant.Text, UnixTimestamp, local_timestamp);

                System.IO.File.AppendAllText(trialsavingpath, t_str);

            }
            
            return true;
        }
        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            if (bt_enter.Enabled)
            {
                var str_score = String.Format("\"{{1:{0},2:{1},3:{2},4:{3}}}\"", textBox_score_0.Text, textBox_score_1.Text, textBox_score_2.Text, textBox_score_3.Text);

                //t_str = String.Format("{0},Trial,{1},END,{2},{3},{4},\"{5}\"\r\n", textBox_participant.Text, trialIndex.Text, textBox_score_0.Text, UnixTimestamp, local_timestamp, textBox_comment.Text.Replace("\r\n"," "));
                //string t_str = String.Format("{0},{1},{2},\"{3}\"\r\n", str_trial, label_time.Text, str_score, textBox_comment.Text.Replace("\r\n", " "));

                string t_str = String.Format("{{{0}" +
                        "\"duration\":\"{1}\"," +
                        "\"score\":\"{2}\"," +
                        "\"comment\":\"{3}\"}},\r\n",
                        str_trial, label_time.Text, str_score, textBox_comment.Text.Replace("\r\n", " "));

                System.IO.File.AppendAllText(trialsavingpath, t_str);

            }
            if (button_endtask.Enabled)
            {
                var local_timestamp = DateTimeOffset.Now.ToString("MM/dd/yyyy hh:mm:ss.fff").ToString();
                var UnixTimestamp = new DateTimeOffset(DateTime.UtcNow).ToUnixTimeMilliseconds();

                if (System.IO.File.Exists(trialsavingpath))
                {
                    string t_str = String.Format("{{\"participant\":\"{0}\"," +
                        "\"unixtimestamp\":\"{1}\"," +
                        "\"localtimestamp\":\"{2}\"," +
                        "\"recordtype\":\"SESSION\"," +
                        "\"status\":\"STOP\"," +
                        "\"taskindex\":\"NA\"," +
                        "\"trialindex\":\"NA\"," +
                        "\"duration\":\"NA\"," +
                        "\"score\":\"NA\"," +
                        "\"comment\":\"NA\"}}\r\n",
                        textBox_participant.Text, UnixTimestamp, local_timestamp);

                    //participant,unixtimestamp,localtimestamp,recordtype,status,taskindex,trialindex,duration,score,comment
                    // t_str = String.Format("{0},{1},{2},SESSION,STOP,NA,NA,NA,NA,NA\r\n", textBox_participant.Text, UnixTimestamp, local_timestamp);
                    System.IO.File.AppendAllText(trialsavingpath, t_str);

                    t_str = "}\r\n";
                    System.IO.File.AppendAllText(trialsavingpath, t_str);

                }
            }
            aTimer.Stop();
            aTimer.Dispose();
            if (videoSource == null)
            { return; }
            if (videoSource.IsRunning)
            {
                this.videoSource.Stop();

            }            
            
        }

        private void comboBox1_SelectedIndexChanged(object sender, EventArgs e)
        {
            label1_score_0.Text = task_performance[comboBox1.SelectedIndex, 0];
            label1_score_1.Text = task_performance[comboBox1.SelectedIndex, 1];
            label1_score_2.Text = task_performance[comboBox1.SelectedIndex, 2];
            label1_score_3.Text = task_performance[comboBox1.SelectedIndex, 3];
            //open the FormCamera, display the realtime image on the picturebox

        }

        private void Form1_Load(object sender, EventArgs e)
        {
            getCamList();
            addTaskList();
            SetTimer();
            //setup empatica device name
            ReadInitiatefile("virtualcoach.ini");
        }

        private void ReadInitiatefile(string filename)
        {
            //E4Wristband:AB2B64
            //E4Wristband: 3A4FCD
            if (System.IO.File.Exists(filename))
            {
                using (StreamReader reader = new StreamReader(filename))
                {
                    List<string> lines = new List<string>();
                    List<string> t_subscribe = new List<string>();
                    int t_sleep = 1000;
                    while (!reader.EndOfStream)
                    {
                        string t_str=reader.ReadLine();
                        if (t_str.Contains("E4Wristband:"))
                        {
                            lines.Add(t_str.Replace("E4Wristband:", ""));
                        }
                        else if (t_str.Contains("E4Sleeptime:"))
                        {
                            t_sleep = Convert.ToInt32(t_str.Replace("E4Sleeptime:", ""));
                        }
                        else if (t_str.Contains("E4Subscribe:"))
                        {
                            t_subscribe.Add(t_str.Replace("E4Subscribe:", "device_subscribe ") +" ON");
                        }
                    }
                    if (lines.Count == 1)
                    {
                        checkBox_empatica_0.Text = lines[0];
                    }
                    else if (lines.Count > 1)
                    {
                        checkBox_empatica_0.Text = lines[0];
                        checkBox_empatica_1.Text = lines[1];
                    }
                    m_empatica_0.SetupEmpaticaDevice(t_subscribe.ToArray(), t_sleep);
                    m_empatica_1.SetupEmpaticaDevice(t_subscribe.ToArray(), t_sleep);
                }
            }

        }

        private void SetTimer()
        {
            // Create a timer with a two second interval.
            aTimer = new System.Timers.Timer(100);
            // Hook up the Elapsed event for the timer. 
            aTimer.Elapsed += OnTimedEvent;
            aTimer.AutoReset = true;
            aTimer.Enabled = true;
        }
        public byte[] imgToByteArray(Image img)
        {
            using (MemoryStream mStream = new MemoryStream())
            {
                img.Save(mStream, img.RawFormat);
                return mStream.ToArray();
            }
        }
        public static void SendToModel(string folder, int index)
        {            
            try
            {
                var t_time= DateTime.Now;
                string filename = String.Format("{0}/{1}.bmp", folder, index);
                var wb = new WebClient();
                var response = wb.UploadFile(url, filename);
                string responseInString = Encoding.UTF8.GetString(response);
            
                m_sortedlist.Add(index , responseInString);
                //File.Delete(filename);
                //var snap = DateTime.Now - t_time;
                Console.WriteLine(String.Format("{0}: {1}, response time: {2}", index, responseInString, DateTime.Now - t_time));
            }
            catch
            {
                //check here why it failed and ask user to retry if the file is in use.
                Console.WriteLine(String.Format("Catch---------------------{0}", index));
                return ;
            }
            return ;
        }


        public static async Task SendAliveMessageAsync(string folder, int index)
        {
            // SendToModel(folder, index);
            /*
            var task1 = new Task(()=> {
                SendToModel(folder, index);
            });
            task1.Start();
            */
            var task2 = Task.Factory.StartNew(() =>
            {
                SendToModel(folder, index);
            });
        }


        private void OnTimedEvent(Object source, ElapsedEventArgs e)
        {
            //Console.WriteLine("The Elapsed event was raised at {0:HH:mm:ss.fff}",e.SignalTime);
            if (m_starttimer && aTimer.Enabled)
            {
                string folder;
                int index;
                int lastItem=0;
                int divisor = 5;
                this.Invoke((MethodInvoker)delegate () {
                                    
                    folder = m_cameras[comboBox_showcameras.SelectedIndex].Getimagefolder();
                    index = m_cameras[comboBox_showcameras.SelectedIndex].Getframeindex();

                    // get the last key of m_sortedlist
                    if (m_sortedlist.Count > 0)
                    {
                        lastItem = m_sortedlist.Keys.Last();
                    }
                    
                    
                    // if the last key is too far from the index then stop sending

                    if (index % divisor == 0)
                    {
                        divisor = (index - lastItem > 15) ? 10 : 5;
                        SendAliveMessageAsync(folder, index - 2);
                        // draw on canvas
                        Console.WriteLine(String.Format("index {0}, list {1}, lastindex {2}, divisor {3}", index, m_sortedlist.Count, lastItem, divisor));
                        if (m_sortedlist.Count > 0)
                        {
                            var tvalue = m_sortedlist.Values.Last().Substring(8).Replace("]}", "\r").Split(',');

                            m_x = Int32.Parse(tvalue[0]);
                            m_y = Int32.Parse(tvalue[1]);
                        }                        
                    }
                    // method 1: draw on graphics
                    Image img = new Bitmap(m_cameras[comboBox_showcameras.SelectedIndex].GetPicture());
                    Graphics gf = Graphics.FromImage(img);
                    gf.DrawRectangle(Pens.Green, m_x-50, m_y - 50, 100, 100);
                    pictureBox1.Image = img;

                    // method 2
                    //pictureBox1.Image = m_cameras[comboBox_showcameras.SelectedIndex].GetPicture();

                });
                                
            }
            
    }

    private void addTaskList()
    {



        foreach ( string item in task_list)
        {
            comboBox1.Items.Add(item);

        }
        comboBox1.SelectedIndex = 0;

    }

    private void show_Click(object sender, EventArgs e)
    {
        //create forms and show
        comboBox_showcameras.Items.Clear();
        int t_count = 0;
        m_imagesavecount = 0;
        foreach (ListViewItem item in listView_CameraControl.Items)
        {
            if (item.Checked)
            {
                //if normal then show normal,
                //else show FlirFileFormat
                if (item.SubItems[1].Text.Contains("FLIR") && checkBox_thermalapi.Checked)//&& checkBox_thermalapi.Checked
                {
                    MainWindow t_cameraForm = new MainWindow();
                    t_cameraForm.m_index = item.Index;

                    t_cameraForm.Show();
                    m_thermalcams.Add(t_cameraForm);
                }
                else
                {
                    FormCameras t_cameraForm = new FormCameras();
                    t_cameraForm.SetDeviceMonikerString(videoDevices[item.Index].MonikerString);
                    t_cameraForm.m_index = item.Index;

                    t_cameraForm.Show();
                    m_cameras.Add(t_cameraForm);
                    comboBox_showcameras.Items.Add(item.SubItems[1].Text + "_" + item.Index);
                    //comboBox_showcameras.SelectedIndex = 0;
                }                    

                //add cameras into the comboBox_showcameras
                t_count++;
            }

        }


        if (t_count>0)
        {
            //_cascadeClassifier = new CascadeClassifier(@"..\data\haarcascades\haarcascade_frontalface_alt2.xml");
        }


    }

    private void bt_empatica_Click(object sender, EventArgs e)
    {
        //empaticaForm = new Form_Empatica();
        //empaticaForm.m_empaticadevicename = textBox_empatica.Text;
        //empaticaForm.Show();


        if (!m_empaticarunning)
        {
            try
            {
                if (checkBox_empatica_0.Checked)
                {
                    if (m_empatica_0.StartClient(checkBox_empatica_0.Text))
                    {
                        checkBox_empatica_0.Enabled = false;
                    }
                    else
                    {
                        checkBox_empatica_0.Checked = false;
                    }
                }
                if (checkBox_empatica_1.Checked)
                {
                    if (m_empatica_1.StartClient(checkBox_empatica_1.Text))
                    {
                        checkBox_empatica_1.Enabled = false;
                    }
                    else
                    {
                        checkBox_empatica_1.Checked = false;
                    }
                }

                bt_empatica.Text = "Stop Empatica";
            }
            catch
            {
                Console.WriteLine("Empatica is not running correctly");
                return;
            }

        }
        else
        {
            if (checkBox_empatica_0.Checked)
            {
                m_empatica_0.StopClient();
                checkBox_empatica_0.Enabled = true;
            }
            if (checkBox_empatica_1.Checked)
            {
                m_empatica_1.StopClient();
                checkBox_empatica_1.Enabled = true;
            }

            bt_empatica.Text = "Start Empatica";
        }
        m_empaticarunning = !m_empaticarunning;


    }

    private void comboBox_showcameras_SelectedIndexChanged(object sender, EventArgs e)
    {
        //pictureBox1.Image = m_cameras[comboBox_showcameras.SelectedIndex].GetPicture();
        m_starttimer = true;
        //pictureBox1.Image = m_cameras[comboBox_showcameras.SelectedIndex].videoimg;
    }

    private void timer_empatica_Tick(object sender, EventArgs e)
    {
        if (startsession)
        {
            if (checkBox_empatica_0.Checked)
            {
                m_empatica_0.SavingEverySecond();
            }
            if (checkBox_empatica_1.Checked)
            {
                m_empatica_1.SavingEverySecond();
            }

        }
    }

    private void checkBox_face_CheckedChanged(object sender, EventArgs e)
    {

    }



    private void bt_enter_Click(object sender, EventArgs e)
    {
        //task_list[comboBox1.SelectedIndex], trialIndex.Text, textBox_score_0.Text,
        //trial format: participant,unixtimestamp,localtimestamp,recordtype,status,
        //taskindex,trialindex,score,comment

        var str_score = String.Format("\"{{1:{0},2:{1},3:{2},4:{3}}}\"", textBox_score_0.Text, textBox_score_1.Text, textBox_score_2.Text, textBox_score_3.Text);

        //t_str = String.Format("{0},Trial,{1},END,{2},{3},{4},\"{5}\"\r\n", textBox_participant.Text, trialIndex.Text, textBox_score_0.Text, UnixTimestamp, local_timestamp, textBox_comment.Text.Replace("\r\n"," "));
        //string t_str = String.Format("{0},{1},{2},\"{3}\"\r\n", str_trial, label_time.Text, str_score, textBox_comment.Text.Replace("\r\n", " "));

        string t_str = String.Format("{{{0}"+
                "\"duration\":\"{1}\"," +
                "\"score\":\"{2}\"," +
                "\"comment\":\"{3}\"}},\r\n",
                str_trial, label_time.Text, str_score, textBox_comment.Text.Replace("\r\n", " "));


        System.IO.File.AppendAllText(trialsavingpath, t_str);

        textBox_comment.Text = "";
        bt_enter.Enabled = false;


        textBox_score_0.Text = "0";
        textBox_score_1.Text = "0";
        textBox_score_2.Text = "0";
        textBox_score_3.Text = "0";

        label_time.Text = "00:00";
        m_trialstart = DateTime.Now;//trialperiod = 0;
        trialIndex.Enabled = true;
        comboBox1.Enabled = true;
        bt_trial.Enabled = true;

    }

    private void button1_Click(object sender, EventArgs e)
    {
        MainWindow t_cameraForm = new MainWindow();
        t_cameraForm.Show();
    }

    private void button_endtask_Click(object sender, EventArgs e)
    {

        if (!b_trial_locked)
        {
            MessageBox.Show("Finish the running trial!");
            return;
        }
        if (bt_enter.Enabled)
        {
            MessageBox.Show("Enter the previous score!");
            return;
        }

        m_starttimer = false;
        startsession = false;

        //loggging
        var local_timestamp = DateTimeOffset.Now.ToString("MM/dd/yyyy hh:mm:ss.fff").ToString();
        var UnixTimestamp = new DateTimeOffset(DateTime.UtcNow).ToUnixTimeMilliseconds();

        if (System.IO.File.Exists(trialsavingpath))
        {
            string t_str = String.Format("{{\"participant\":\"{0}\"," +
                "\"unixtimestamp\":\"{1}\"," +
                "\"localtimestamp\":\"{2}\"," +
                "\"recordtype\":\"SESSION\"," +
                "\"status\":\"STOP\"," +
                "\"taskindex\":\"NA\"," +
                "\"trialindex\":\"NA\"," +
                "\"duration\":\"NA\"," +
                "\"score\":\"NA\"," +
                "\"comment\":\"NA\"}}\r\n",
                textBox_participant.Text, UnixTimestamp, local_timestamp);

            //participant,unixtimestamp,localtimestamp,recordtype,status,taskindex,trialindex,duration,score,comment
            // t_str = String.Format("{0},{1},{2},SESSION,STOP,NA,NA,NA,NA,NA\r\n", textBox_participant.Text, UnixTimestamp, local_timestamp);
            System.IO.File.AppendAllText(trialsavingpath, t_str);

            t_str = "}\r\n";
            System.IO.File.AppendAllText(trialsavingpath, t_str);

        }            

        button_endtask.Enabled = false;

        textBox_participant.Enabled = false;
        comboBox1.Enabled = false;
        trialIndex.Enabled = false;
        bt_trial.Enabled = false;
        textBox_score_0.Enabled = false;
        textBox_score_1.Enabled = false;
        textBox_score_2.Enabled = false;
        textBox_score_3.Enabled = false;
        textBox_comment.Enabled = false;
        bt_enter.Enabled = false;
        save.Enabled = true;
    }

    private void listView_CameraControl_SelectedIndexChanged(object sender, EventArgs e)
    {

    }

    private void bt_trial_Click(object sender, EventArgs e)
    {
        m_imagesavecount = 0;
        if (!startsession)
        {
            MessageBox.Show("Start session first!");
            return;
        }

        if (bt_enter.Enabled)
        {
            MessageBox.Show("Enter the previous score!");
            return;
        }

        var local_timestamp = DateTimeOffset.Now.ToString("MM/dd/yyyy hh:mm:ss.fff").ToString();
        var UnixTimestamp = new DateTimeOffset(DateTime.UtcNow).ToUnixTimeMilliseconds();

        if (b_trial_locked)
        {

            string trialinfo= String.Format("task{0}_trial{1}_{2}", task_list_log[comboBox1.SelectedIndex], trialIndex.Text, UnixTimestamp);
            foreach (FormCameras item in m_cameras)
            {
                string filepath = String.Format("{0}\\Camera{1}_{2}.avi", m_savingfolder, item.m_index, trialinfo);
                item.StartRecording(filepath);                           
            }
            foreach (MainWindow item in m_thermalcams)
            {
                string filepath = String.Format("{0}\\Thermal{1}_{2}.avi", m_savingfolder, item.m_index, trialinfo);
                item.StartRecording(filepath);                    
            }


            //wait until all opened
            /*
            int t_count = m_thermalcams.Count + m_cameras.Count;
            while (t_count==0)
            {
                foreach (FormCameras item in m_cameras)
                {
                    while (item.CheckFileWriterOpen())
                    {
                        t_count--;
                        System.Threading.Thread.Sleep(500);
                    }

                }
                foreach (MainWindow item in m_thermalcams)
                {
                    while (item.CheckFileWriterOpen())
                    {
                        t_count--;
                        System.Threading.Thread.Sleep(500);
                    }
                }

            }
            */

                //testing
                //m_empaticarunning = true;
                if (m_empaticarunning)
                {
                    if (checkBox_empatica_0.Checked)
                    {
                        m_empatica_0.SavingRecord(m_savingfolder, trialinfo);
                    }
                    if (checkBox_empatica_1.Checked)
                    {
                        m_empatica_1.SavingRecord(m_savingfolder, trialinfo);
                    }

                    timer_empatica.Enabled = true;
                }
                //m_empaticarunning = false;

                gazedatasavingpath = String.Format("{0}\\Tobii_{1}.txt", m_savingfolder,  trialinfo); //m_savingfolder + "_Tobii.txt";

                if (!System.IO.File.Exists(gazedatasavingpath))
                {
                    //create file
                    using (var t_file = System.IO.File.Create(gazedatasavingpath)) ;
                }

                local_timestamp = DateTimeOffset.Now.ToString("MM/dd/yyyy hh:mm:ss.fff").ToString();
                UnixTimestamp = new DateTimeOffset(DateTime.UtcNow).ToUnixTimeMilliseconds();

                System.IO.File.WriteAllText(gazedatasavingpath, "DEVICE,X,Y,Z,LPDA_X,LPDA_Y,RPDA_X,RPDA_Y,Pupil_VA,Pupil_left,Pupil_right,UnixTS,TimeStamp\r\n");

                eyetrackingrecordenabled = true;
                string t_str = String.Format("{{\"participant\":\"{0}\"," +
                    "\"unixtimestamp\":\"{1}\"," +
                    "\"localtimestamp\":\"{2}\"," +
                    "\"recordtype\":\"TRIAL\"," +
                    "\"status\":\"START\"," +
                    "\"taskindex\":\"{3}\"," +
                    "\"trialindex\":\"{4}\"," +
                    "\"duration\":\"NA\"," +
                    "\"score\":\"NA\"," +
                    "\"comment\":\"NA\"}},\r\n",
                    textBox_participant.Text, UnixTimestamp, local_timestamp, task_list_log[comboBox1.SelectedIndex], trialIndex.Text);
                                
                bt_trial.Text = "End the Trial " + trialIndex.Text;
                bt_enter.Enabled = false;
                trialIndex.Enabled = false;
                comboBox1.Enabled = false;

                System.IO.File.AppendAllText(trialsavingpath, t_str);

                System.Threading.Thread.Sleep(500);

                m_trialstart = DateTime.Now;
                timer_main.Enabled = true;
            }
            else
            {
                foreach (FormCameras item in m_cameras)
                {
                    item.StopRecording();
                }
                foreach (MainWindow item in m_thermalcams)
                {
                    item.StopRecording();
                }
                timer_empatica.Enabled = false;
                eyetrackingrecordenabled = false;
                //write end timestamp into the file
                //str_trial = String.Format("{0},{1},{2},Trial,END,{3},{4}", textBox_participant.Text, UnixTimestamp, local_timestamp, task_list_log[comboBox1.SelectedIndex], trialIndex.Text);

                str_trial = String.Format("\"participant\":\"{0}\"," +
                    "\"unixtimestamp\":\"{1}\"," +
                    "\"localtimestamp\":\"{2}\"," +
                    "\"recordtype\":\"TRIAL\"," +
                    "\"status\":\"END\"," +
                    "\"taskindex\":\"{3}\"," +
                    "\"trialindex\":\"{4}\","
                    , textBox_participant.Text, UnixTimestamp, local_timestamp, task_list_log[comboBox1.SelectedIndex], trialIndex.Text);


                bt_trial.Text = "Start A New Trial";
                

                timer_main.Enabled = false;

                bt_enter.Enabled = true;
                bt_trial.Enabled = false;

            }
            //save video
            //iterate the listcameras
            
            
            b_trial_locked = !b_trial_locked;
        }

        private void button3_Click(object sender, EventArgs e)
        {
            m_eyetrackerfrequency = float.Parse(textBox_pupil.Text);
        }
       
        private void button4_Click(object sender, EventArgs e)
        {
            //send http post request
            OpenFileDialog openFileDialog = new OpenFileDialog();
            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                var filename = openFileDialog.FileName;
                using (var wb = new WebClient())
                {

                    //send imagefile app_ori.py
                    var timenow = DateTime.Now;
                    var response = wb.UploadFile(url, filename);
                    string responseInString = Encoding.UTF8.GetString(response);

                    var timespan = DateTime.Now.Subtract(timenow);
                    Console.WriteLine(String.Format("Response time {0}:{1}", timespan,responseInString));
                    
                    //send imagebyte
                    
                }
            }
        }

        class CallEyeTrackerManager
        {
            internal static void Execute(IEyeTracker eyeTracker)
            {
                if (eyeTracker != null)
                {
                    CallEyeTrackerManagerExample(eyeTracker);
                }
            }
            // <BeginExample>
            private static void CallEyeTrackerManagerExample(IEyeTracker eyeTracker)
            {
                string etmStartupMode = "usercalibration";// "usercalibration";// "displayarea";//"--version"
                string etmBasePath = Path.GetFullPath(Path.Combine(Environment.GetEnvironmentVariable("LocalAppData"),
                                                                    "TobiiProEyeTrackerManager"));
                string appFolder = Directory.EnumerateDirectories(etmBasePath, "app*").FirstOrDefault();
                string executablePath = Path.GetFullPath(Path.Combine(etmBasePath,
                                                                        appFolder,
                                                                        "TobiiProEyeTrackerManager.exe"));
                string arguments = "--device-address=" + eyeTracker.Address + " --mode=" + etmStartupMode;
                try
                {
                    Process etmProcess = new Process();
                    // Redirect the output stream of the child process.
                    etmProcess.StartInfo.UseShellExecute = false;
                    etmProcess.StartInfo.RedirectStandardError = true;
                    etmProcess.StartInfo.RedirectStandardOutput = true;
                    etmProcess.StartInfo.FileName = executablePath;
                    etmProcess.StartInfo.Arguments = arguments;
                    etmProcess.Start();
                    string stdOutput = etmProcess.StandardOutput.ReadToEnd();

                    etmProcess.WaitForExit();
                    int exitCode = etmProcess.ExitCode;
                    if (exitCode == 0)
                    {
                        Console.WriteLine("Eye Tracker Manager was called successfully!");
                    }
                    else
                    {
                        Console.WriteLine("Eye Tracker Manager call returned the error code: {0}", exitCode);
                        foreach (string line in stdOutput.Split(Environment.NewLine.ToCharArray()))
                        {
                            if (line.StartsWith("ETM Error:"))
                            {
                                Console.WriteLine(line);
                            }
                        }
                    }
                }
                catch (Exception e)
                {
                    Console.WriteLine(e.Message);
                }
            }
            // <EndExample>
        }

    }
}

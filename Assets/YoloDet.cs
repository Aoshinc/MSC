using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.DnnModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityUtils;
using OpenCVForUnity.UnityUtils.Helper;
using Image = UnityEngine.UI.Image;

[RequireComponent(typeof(WebCamTextureToMatHelper))]
public class YoloTest : MonoBehaviour
{
    public Image warningSign;
    public RawImage rawImage;
    public Canvas canvas;
    public AudioSource warningAudioSource;

    public float confThreshold;

    public float nmsThreshold;

    Texture2D texture;

    WebCamTextureToMatHelper webCamTextureToMatHelper;

    Mat rgbMat;

    Net net;

    List<string> classNames;
    List<string> outBlobNames;
    List<string> outBlobTypes;

    string modelCfg;
    string modelWeights;
    string modelClasses;
    float warningSignWidth;
    float warningSignHeight;

    private float personDetectedTime = 0; 
    private float warningDuration = 2f; 

    // Use this for initialization
    void Start()
    {
        // Get all device names of cameras
        WebCamDevice[] devices = WebCamTexture.devices;

        // List the camera names
        for (int i = 0; i < devices.Length; i++)
        {
            Debug.Log("Camera name: " + devices[i].name);
        }

        warningSignWidth = warningSign.rectTransform.sizeDelta.x;
        warningSignHeight = warningSign.rectTransform.sizeDelta.y;
        webCamTextureToMatHelper = gameObject.GetComponent<WebCamTextureToMatHelper>();

        modelClasses = Utils.getFilePath("dnn/coco.names");
        modelCfg = Utils.getFilePath("dnn/yolov3-tiny.cfg");
        modelWeights = Utils.getFilePath("dnn/yolov3-tiny.weights");
        Init();
        rawImage.texture = texture;

    }

    // Use this for initialization
    void Init()
    {
        net = Dnn.readNet(modelWeights, modelCfg);
        outBlobNames = getOutputsNames(net);
        outBlobTypes = getOutputsTypes(net);
        webCamTextureToMatHelper.Initialize();
        Debug.Log("OnWebCamTextureToMatHelperInitialized");
        Mat webCamTextureMat = webCamTextureToMatHelper.GetMat();
        texture = new Texture2D(webCamTextureMat.cols(), webCamTextureMat.rows(), TextureFormat.RGBA32, false);
        if (gameObject.GetComponent<Renderer>())
        {
            gameObject.GetComponent<Renderer>().material.mainTexture = texture;
        }
        gameObject.transform.localScale = new Vector3(webCamTextureMat.cols(), webCamTextureMat.rows(), 1);
        float width = webCamTextureMat.width();
        float height = webCamTextureMat.height();
        float widthScale = (float)Screen.width / width;
        float heightScale = (float)Screen.height / height;
        if (widthScale < heightScale)
        {
            Camera.main.orthographicSize = (width * (float)Screen.height / (float)Screen.width) / 2;
        }
        else
        {
            Camera.main.orthographicSize = height / 2;
        }

        rgbMat = new Mat(webCamTextureMat.rows(), webCamTextureMat.cols(), CvType.CV_8UC3);
    }

    // Update is called once per frame
    void Update()
    {
        if (webCamTextureToMatHelper.IsPlaying() && webCamTextureToMatHelper.DidUpdateThisFrame())
        {
            Mat rgbaMat = webCamTextureToMatHelper.GetMat();
            if (net == null)
            {
                Debug.Log("Yolo model loading fail");
            }
            else
            {
                Imgproc.cvtColor(rgbaMat, rgbMat, Imgproc.COLOR_RGBA2RGB);

                Mat blob = Dnn.blobFromImage(rgbMat, 1 / 255.0, new Size(416, 416), new Scalar(0, 0, 0), false, false);
                // Run a model.
                net.setInput(blob);
                TickMeter tm = new TickMeter();
                tm.start();
                List<Mat> outs = new List<Mat>();
                net.forward(outs, outBlobNames);
                tm.stop();
                PostProcess(rgbaMat, outs, net);
                for (int i = 0; i < outs.Count; i++)
                {
                    outs[i].Dispose();
                }
                blob.Dispose();
            }
            Utils.fastMatToTexture2D(rgbaMat, texture);
            rawImage.texture = texture;
        }
    }
    void OnDestroy()
    {
        if(webCamTextureToMatHelper != null)
            webCamTextureToMatHelper.Dispose();

        if (net != null)
            net.Dispose();

        if (rgbMat != null)
            rgbMat.Dispose();

        if (texture != null)
        {
            Texture2D.Destroy(texture);
            texture = null;
        }
    }

    private void PostProcess(Mat frame, List<Mat> outs, Net net)
    {
        RectTransform canvasRect = canvas.GetComponent<RectTransform>();

        string outLayerType = outBlobTypes[0];

        List<int> classIdsList = new List<int>();
        List<float> confidencesList = new List<float>();
        List<OpenCVForUnity.CoreModule.Rect> boxesList = new List<OpenCVForUnity.CoreModule.Rect>();
        List<Vector2> boxesCenterList = new List<Vector2>();
        if (outLayerType == "Region")
        {
            for (int i = 0; i < outs.Count; ++i)
            {

                float[] objPosition = new float[5];
                float[] confidenceData = new float[outs[i].cols() - 5];

                for (int p = 0; p < outs[i].rows(); p++)
                {                  
                    outs[i].get(p, 0, objPosition);
                    outs[i].get(p, 5, confidenceData);
                    float maxConfidence = float.MinValue; 
                    int maxIdx = -1; 
                    for (int j = 0; j < confidenceData.Length; j++)
                    {
                        if (confidenceData[j] > maxConfidence)
                        {
                            maxConfidence = confidenceData[j];
                            maxIdx = j;
                        }
                    }
                    if(maxIdx == 0)
                    {
                        float confidence = confidenceData[maxIdx];

                        if (confidence > confThreshold)
                        {
                            int centerX = (int)(objPosition[0] * frame.cols());
                            int centerY = (int)(objPosition[1] * frame.rows());
                            int width = (int)(objPosition[2] * frame.cols());
                            int height = (int)(objPosition[3] * frame.rows());
                            int left = centerX - width / 2;
                            int top = centerY - height / 2;
                            classIdsList.Add(maxIdx);
                            confidencesList.Add((float)confidence);
                            boxesList.Add(new OpenCVForUnity.CoreModule.Rect(left, top, width, height));
                            boxesCenterList.Add(new Vector2(centerX, centerY));
                        }
                    }
                }
            }
        }
        else
        {
            Debug.Log("Unknown output layer type");
        }
        MatOfRect boxes = new MatOfRect();
        boxes.fromList(boxesList);

        MatOfFloat confidences = new MatOfFloat();
        confidences.fromList(confidencesList);

        MatOfInt indices = new MatOfInt();
        Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

        if(indices.total() != 0)
        {
            int maximumIndex = 0;
            double maximumArea = 0;
            personDetectedTime = Time.time;
            warningSign.enabled = true;
            rawImage.enabled = true;
            for (int i = 0; i < indices.total(); ++i)
            {
                int idx = (int)indices.get(i, 0)[0];
                OpenCVForUnity.CoreModule.Rect box = boxesList[idx];
                double squdArea = box.area();
                if (squdArea > maximumArea)
                {
                    maximumArea = squdArea;
                    maximumIndex = idx;
                }
                Imgproc.rectangle(frame, new Point(box.x, box.y), new Point(box.x + box.width, box.y + box.height), new Scalar(0, 255, 0, 255), 2);
            }
            Vector2 screenCenter = new Vector2(Screen.width / 2, Screen.height / 2);
            Vector2 boxCenter = new Vector2(boxesCenterList[maximumIndex].x * ((float)Screen.width/frame.width()),
                Screen.height - boxesCenterList[maximumIndex].y * ((float)Screen.height / frame.height()));
 
            Vector2 direction = (boxCenter - screenCenter).normalized;
            Vector2 edgePoint;
            // Determine which edge the intersection is on and
            // compare the direction to the aspect ratio of the screen to determine whether it is a left or right edge or an up or down edge
            if (Mathf.Abs(direction.x) * Screen.height > Mathf.Abs(direction.y) * Screen.width)
            {
                if (direction.x > 0)
                {
                    // Right
                    edgePoint = new Vector2(Screen.width - warningSignWidth / 2, screenCenter.y + direction.y / direction.x * (Screen.width - screenCenter.x));
                }
                else
                {
                    // Left
                    edgePoint = new Vector2(warningSignWidth / 2, screenCenter.y - direction.y / direction.x * screenCenter.x);
                }
            }
            else
            {
                if (direction.y > 0)
                {
                    // Top
                    edgePoint = new Vector2(screenCenter.x + direction.x / direction.y * (Screen.height - screenCenter.y), Screen.height - warningSignHeight / 2);
                }
                else
                {
                    // Down
                    edgePoint = new Vector2(screenCenter.x - direction.x / direction.y * screenCenter.y, warningSignHeight / 2);
                }
            }

            Vector2 canvasEdgePoint = ConvertScreenPosToCanvasPos(edgePoint, canvasRect);
            warningSign.transform.localPosition = canvasEdgePoint;


            Debug.DrawLine(screenCenter, boxCenter, Color.red);
            warningSign.transform.position = edgePoint;

            warningAudioSource.maxDistance = Screen.width/2 - warningSignWidth;
            warningAudioSource.transform.position = new Vector3(boxCenter.x - (float)Screen.width/2, 0, 0);
            double areaRatio =  maximumArea / (double)(frame.width() * frame.height());
            if (!warningAudioSource.isPlaying)
            {
                float volume = Mathf.Clamp((float)((areaRatio - 0.2) / (0.6 - 0.2)), 0f, 1f);
                warningAudioSource.volume = volume;
                warningAudioSource.Play();
            }
                
        }
        else
        {
            if (Time.time - personDetectedTime > warningDuration)
            {
                warningSign.enabled = false;
                rawImage.enabled = false;
            }
        }
        indices.Dispose();
        boxes.Dispose();
        confidences.Dispose();


    }

    private List<string> getOutputsNames(Net net)
    {
        List<string> names = new List<string>();


        MatOfInt outLayers = net.getUnconnectedOutLayers();
        for (int i = 0; i < outLayers.total(); ++i)
        {
            names.Add(net.getLayer(new DictValue((int)outLayers.get(i, 0)[0])).get_name());
        }
        outLayers.Dispose();

        return names;
    }
    private List<string> getOutputsTypes(Net net)
    {
        List<string> types = new List<string>();


        MatOfInt outLayers = net.getUnconnectedOutLayers();
        for (int i = 0; i < outLayers.total(); ++i)
        {
            types.Add(net.getLayer(new DictValue((int)outLayers.get(i, 0)[0])).get_type());
        }
        outLayers.Dispose();

        return types;
    }

    Vector2 ConvertScreenPosToCanvasPos(Vector2 screenPos, RectTransform canvasRect)
    {
        Vector2 normalizedScreenPos = new Vector2(screenPos.x / Screen.width, screenPos.y / Screen.height);
        // Map to Canvas coordinates
        Vector2 canvasPos = new Vector2(normalizedScreenPos.x * canvasRect.sizeDelta.x, normalizedScreenPos.y * canvasRect.sizeDelta.y);
        return canvasPos;
    }
}

using UnityEngine;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ObjdetectModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityUtils;
using OpenCVForUnity.UnityUtils.Helper;
using UnityEngine.UI;


[RequireComponent(typeof(WebCamTextureToMatHelper))]

public class BystanderDect : MonoBehaviour
{
    public RawImage rawImage;

    public TMPro.TextMeshProUGUI warningText;

    Mat rgbMat;

    Texture2D texture;

    MatOfRect face;

    WebCamTextureToMatHelper webCamTextureToMatHelper;

    CascadeClassifier cascade;

    HOGDescriptor hog;

    //protected static readonly string LBP_CASCADE_FILENAME = "haarcascade_fullbody.xml";
    protected static readonly string LBP_CASCADE_FILENAME = "haarcascade_frontalface_alt2.xml";
    private float personDetectedTime = 0;     // Used to store the last time a pedestrian was detected
    private float warningDuration = 2f;       // The duration of the warning display
    // Start is called before the first frame update
    void Start()
    {
        webCamTextureToMatHelper = gameObject.GetComponent<WebCamTextureToMatHelper>();

        cascade = new CascadeClassifier();
        cascade.load(Utils.getFilePath(LBP_CASCADE_FILENAME));
        if (cascade.empty())
        {
            Debug.LogError("cascade file is not loaded. Please copy from ¡°OpenCVForUnity/StreamingAssets/¡± to ¡°Assets/StreamingAssets/¡± folder. ");
        }

        webCamTextureToMatHelper.Initialize();
        Debug.Log("OnWebCamTextureToMatHelperInitialized");

        Mat webCamTextureMat = webCamTextureToMatHelper.GetMat();

        texture = new Texture2D(webCamTextureMat.cols(), webCamTextureMat.rows(), TextureFormat.RGB24, false);
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

        rgbMat = new Mat(webCamTextureMat.rows(), webCamTextureMat.cols(), CvType.CV_8UC1);

        face = new MatOfRect();

        rawImage.texture = texture;

        hog = new HOGDescriptor();
    }
    // Update is called once per frame
    void Update()
    {
        if (webCamTextureToMatHelper.IsPlaying() && webCamTextureToMatHelper.DidUpdateThisFrame())
        {

            Mat rgbaMat = webCamTextureToMatHelper.GetMat();

            // Bystander Detection by HOG descriptor
            Imgproc.cvtColor(rgbaMat, rgbMat, Imgproc.COLOR_RGBA2RGB);

            PeopleDetection(rgbMat);
            FaceDetection(rgbMat);

            Utils.fastMatToTexture2D(rgbMat, texture);
            rawImage.texture = texture;
        }
     }
    void OnDestroy()
    {
        Debug.Log("Bystanderdetection disposed");
        if (webCamTextureToMatHelper != null)
            webCamTextureToMatHelper.Dispose();
        if (rgbMat != null)
            rgbMat.Dispose();

        if (texture != null)
        {
            Texture2D.Destroy(texture);
            texture = null;
        }

        if (face != null)
            face.Dispose();

        if (hog != null)
            hog.Dispose();
    }

    void PeopleDetection(Mat rgbMat)
    {
        using MatOfRect locations = new();
        using MatOfDouble weights = new();
        {
            hog.setSVMDetector(HOGDescriptor.getDefaultPeopleDetector());
            hog.detectMultiScale(rgbMat, locations, weights);

            OpenCVForUnity.CoreModule.Rect[] rects = locations.toArray();
            for (int i = 0; i < rects.Length; i++)
            {
                Imgproc.rectangle(rgbMat, new Point(rects[i].x, rects[i].y), new Point(rects[i].x + rects[i].width, rects[i].y + rects[i].height), new Scalar(255, 0, 0), 2);
            }
        }
    }

    void FaceDetection(Mat rgbMat)
    {
        if (cascade != null)
            cascade.detectMultiScale(rgbMat, face, 1.1, 2, 2, 
                new Size(rgbMat.cols() * 0.1, rgbMat.rows() * 0.1), new Size());

        OpenCVForUnity.CoreModule.Rect[] fac_rects = face.toArray();
        for (int i = 0; i < fac_rects.Length; i++)
        {
            Imgproc.rectangle(rgbMat, new Point(fac_rects[i].x, fac_rects[i].y), new Point(fac_rects[i].x + fac_rects[i].width, fac_rects[i].y + fac_rects[i].height), new Scalar(0, 0, 255), 2);       
        }

        if (fac_rects.Length != 0)
        {
            personDetectedTime = Time.time;
            rawImage.enabled = true; // show rawImage when a face is detected
            warningText.text = "Warning: Someone around you";
        }
        else
        {
            if (Time.time - personDetectedTime > warningDuration)
            {
                rawImage.enabled = false; // hide rawImage when no face is detected
                warningText.text = "";
            }
        }
    }
}

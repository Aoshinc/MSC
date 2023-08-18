using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.UIElements;

public class UIBillboard : MonoBehaviour
{
    public Camera userCamera;

    float distance = 1.0f;
    private void Start()
    {
        userCamera = Camera.main;
    }

    private void Update()
    {
        // Make the UI always direct to the camera
        transform.position = userCamera.transform.position + userCamera.transform.forward * distance;

        // Make the Canvas always direct to the camera
        transform.LookAt(2 * transform.position - userCamera.transform.position);
    }
}

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraMan : MonoBehaviour
{

    public Transform transformA;
    public Transform transformB;

    bool isMoving = false;

    bool isTargetA = true;

    void Start()
    {
        transform.position = transformA.position;
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyUp(KeyCode.P)) {
            if (isTargetA)
            {
                // this.transform.position = transformB.position;
                MoveTo(transformB.position);
                this.transform.rotation = transformB.rotation;
            } else {
                MoveTo(transformA.position);
                this.transform.rotation = transformA.rotation;
            }
            isTargetA = !isTargetA;
        }
        
    }
    
    void MoveTo(Vector3 position) {
        if (isMoving) {
            return;
        }
        isMoving = true;
        StartCoroutine(LerpFromTo(transform.position, position, 1f));
        
        IEnumerator LerpFromTo(Vector3 pos1, Vector3 pos2, float duration) {
            for (float t=0f; t<duration; t += Time.deltaTime) {
                transform.position = Vector3.Lerp(pos1, pos2, t / duration);
                yield return 0;
            }
            transform.position = pos2;
            isMoving = false;
        }
    }

}
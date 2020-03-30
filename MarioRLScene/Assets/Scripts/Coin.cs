using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Coin : MonoBehaviour
{

    float originalHeight = 0;

    void Awake()
    {
        originalHeight = transform.position.y;
    }

    void Update()
    {
        transform.Rotate (0, 50*Time.deltaTime, 0);

        Vector3 pos = transform.position;
        float newY = Mathf.Sin(Time.time * 1)/3 + originalHeight;
        transform.position = new Vector3(pos.x, newY, pos.z);
    }

    void OnTriggerEnter(Collider other) {
         if (other.tag == Tags.mario) {
             gameObject.SetActive(false);
         }
     }

}

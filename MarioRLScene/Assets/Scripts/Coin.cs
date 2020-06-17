using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Coin : MonoBehaviour
{

    float originalHeight = 0;
    MeshRenderer m_renderer;

    protected bool shouldAnimateBounce = true;

    void Awake()
    {
        originalHeight = transform.position.y;
        m_renderer = GetComponent<MeshRenderer>();
    }

    protected virtual void Update()
    {
        transform.Rotate (0, 50*Time.deltaTime, 0);

        if (shouldAnimateBounce)
        {
            AnimateBounce();
        }
    }

    protected virtual void OnTriggerEnter(Collider other) {
         if (other.tag == Tags.mario) {
             m_renderer.enabled = false;
         }
     }

    void OnEnable()
    {
        EnvironmentManager.ResetState += Reset;
    }


    void OnDisable()
    {
        EnvironmentManager.ResetState -= Reset;
    }


    protected virtual void Reset()
    {
        m_renderer.enabled = true;
    }


    void AnimateBounce()
    {
        Vector3 pos = transform.position;
        float newY = Mathf.Sin(Time.time * 1)/3 + originalHeight;
        transform.position = new Vector3(pos.x, newY, pos.z);
    }

}

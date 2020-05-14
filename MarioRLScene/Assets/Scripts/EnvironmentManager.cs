using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnvironmentManager : MonoBehaviour
{
    public Environment environmentObject;

    public Environment[] environments;
    const float environmentSpacing = 14;

    public delegate void ResetEvent();
    public static event ResetEvent ResetState;

    // Start is called before the first frame update
    void Start()
    {
        for (int i = 0; i < environments.Length; i++)
        {
            AddEnvironment(i);
            environments[i].Reset();
        }
    }

    void Update()
    {
        if (Input.GetKeyUp(KeyCode.R))
        {
            if (ResetState != null)
            {
                ResetState();
            }
            Reset();
            return;
        }
    }

    void AddEnvironment(int index)
    {
        Environment env = Environment.Instantiate(environmentObject);
        env.id = index;
        env.gameObject.SetActive(true);
        env.transform.position = new Vector3(index * environmentSpacing, 0, 0);
        env.transform.parent = transform;
        environments[index] = env;
    }

    void Reset()
    {
        for (int i = 0; i < environments.Length; i++)
        {
            environments[i].Reset();
        }
    }
}

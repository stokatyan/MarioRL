using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class Environment : MonoBehaviour
{
    public Mario mario;
    public Coin coin;

    public float minX = -4;
    public float maxX = 4;
    public float minZ = -4;
    public float maxZ = 4;

    public delegate void ResetAction();
    public static event ResetAction ResetState;

    void Start()
    {
        Setup();
    }

    void Update()
    {
        if (Input.GetKey(KeyCode.R))
        {
            Reset();
        }
    }

    void Setup()
    {
        Vector3 randomPosition = new Vector3(Random.Range(minX, maxX), 0, Random.Range(minZ, maxZ));
        coin.transform.position = randomPosition;
        randomPosition = new Vector3(Random.Range(minX, maxX), 0, Random.Range(minZ, maxZ));
        float dist = Vector3.Distance(coin.transform.position, randomPosition);
        if (dist < 2)
        {
            if (maxX - coin.transform.position.x > 2)
            {
                randomPosition.x = maxX;
            } else 
            {
                randomPosition.x = minX;
            }
        }
        mario.SetPosition(randomPosition);
    }

    void Reset()
    {
        Setup();
        if (ResetState != null)
        {
            ResetState();
        }
    }

}

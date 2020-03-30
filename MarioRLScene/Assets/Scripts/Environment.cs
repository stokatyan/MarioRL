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

    float updateFrequency = 0.02f;
    float lastUpdateTime = 0;

    public delegate void ResetAction();
    public static event ResetAction ResetState;

    void Start()
    {
        Reset();
    }

    void Update()
    {
        if (Input.GetKeyUp(KeyCode.R))
        {
            Reset();
            return;
        }

        if  (Time.time - lastUpdateTime > updateFrequency) 
        {
            WriteDistance();
            SetAgentAction();
            lastUpdateTime = Time.time;
        }
        
    }

    void Setup()
    {
        Vector3 randomPosition = new Vector3(Random.Range(minX, maxX), 0, Random.Range(minZ, maxZ));
        coin.transform.position = randomPosition;
        randomPosition = new Vector3(Random.Range(minX, maxX), 0, Random.Range(minZ, maxZ));
        float distance = Vector3.Distance(coin.transform.position, randomPosition);
        if (distance < 2)
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
        Pipeline.ClearAction();
        Setup();
        if (ResetState != null)
        {
            ResetState();
        }
    }

    #region I/O

    void SetAgentAction()
    {
        Action action = Pipeline.ReadAction();
        mario.currentAction = action;
    }

    void WriteDistance()
    {
        Vector2 v0 = new Vector2(coin.transform.position.x, coin.transform.position.z);
        Vector2 v1 = new Vector2(mario.transform.position.x, mario.transform.position.z);
        float distance = Vector2.Distance(v0, v1);
        Observation obs = new Observation(distance);

        Pipeline.WriteObservation(obs);
    }

    #endregion

}

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

    float updateFrequency = 0.1f;
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
            SetAgentAction();
            HandleGameState();
            WriteObservation();
            
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
        Pipeline.WriteGameStarted();
    }

    #region I/O

    void HandleGameState()
    {
        bool isGameOver = Pipeline.ReadIsGameOver();
        if (isGameOver)
        {
            Reset();
        }
    }

    void SetAgentAction()
    {
        Action action = Pipeline.ReadAction();
        mario.currentAction = action;
    }

    void WriteObservation()
    {
        Vector2 coinVector = new Vector2(coin.transform.position.x, coin.transform.position.z);
        Vector2 marioVector = new Vector2(mario.transform.position.x, mario.transform.position.z);
        float distance = Vector2.Distance(coinVector, marioVector);

        float[] marioPosition = {marioVector.x, marioVector.y};
        float[] coinPosition = {coinVector.x, coinVector.y};

        Observation obs = new Observation(distance, marioPosition, coinPosition);

        Pipeline.WriteObservation(obs);
    }

    #endregion

}

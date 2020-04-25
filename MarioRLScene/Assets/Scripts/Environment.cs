using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class Environment : MonoBehaviour
{
    public Coin coin;
    public Mario mario;
    public SmallCoin smallCoin;

    public float minX = -4;
    public float maxX = 4;
    public float minZ = -4;
    public float maxZ = 4;
    int consecutiveEvalsCount = 0;

    const int maxSmallCoinCount = 10;
    const int maxEvalSmallCoinCount = 10;
    const float smallCoinFixedY = 1.25f;
    int smallCoinsCollectedCount = 0;

    float updateFrequency = 0.1f;
    float lastUpdateTime = 0;

    public float[] marioYPositions = {-4, -2, 0, 2, 4};
    public Transform[] coinPositions;

    public delegate void ResetEvent();
    public static event ResetEvent ResetState;

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

    void OnEnable()
    {
        SmallCoin.Collected += CollectedSmallCoin;
    }


    void OnDisable()
    {
        SmallCoin.Collected -= CollectedSmallCoin;
    }

    #region Events

    void CollectedSmallCoin()
    {
        smallCoinsCollectedCount += 1;
    }

    #endregion

    #region Setup

    Vector3 CreateRandomPosition()
    {
        return new Vector3(Random.Range(minX, maxX), 0, Random.Range(minZ, maxZ));
    }

    void Setup()
    {
        Vector3 randomPosition = CreateRandomPosition();
        mario.SetPosition(randomPosition);

        consecutiveEvalsCount = 0;

        for (int i = 0; i < maxSmallCoinCount ; i++)
        {
            randomPosition = CreateRandomPosition();
            randomPosition.y = smallCoinFixedY;
            float distance = Vector3.Distance(mario.transform.position, randomPosition);
            if (distance < 2)
            {
                continue;
            }

            SmallCoin sc = SmallCoin.Instantiate(smallCoin);
            sc.gameObject.SetActive(true);
            sc.transform.position = randomPosition;
        }
    }

    void SetupEval()
    {
        Vector3 randomPosition = CreateRandomPosition();
        randomPosition.z = marioYPositions[consecutiveEvalsCount];
        randomPosition.x = minX;
        mario.SetPosition(randomPosition);

        consecutiveEvalsCount += 1;
        if (consecutiveEvalsCount >= marioYPositions.Length)
        {
            consecutiveEvalsCount = 0;
        }

        for (int i = 0; i < coinPositions.Length ; i++)
        {
            SmallCoin sc = SmallCoin.Instantiate(smallCoin);
            sc.gameObject.SetActive(true);
            sc.transform.position = coinPositions[i].position;
        }
    }

    void Reset()
    {
        Pipeline.ClearAction();

        smallCoinsCollectedCount = 0;
        if (ResetState != null)
        {
            ResetState();
        }
        Setup();
        
        Pipeline.WriteGameStarted();
    }

    void ResetEval()
    {
        Pipeline.ClearAction();

        smallCoinsCollectedCount = 0;
        if (ResetState != null)
        {
            ResetState();
        }
        SetupEval();
        
        Pipeline.WriteGameStarted();
    }

    #endregion

    #region I/O

    void HandleGameState()
    {
        int isGameOver = Pipeline.ReadIsGameOver();
        if (isGameOver == GameState.isGameover)
        {
            Reset();
        } else if (isGameOver == GameState.isEvalGameover)
        {
            ResetEval();
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
        float distance = mario.GetNearestCoinDistance();
    
        float[] marioDistances = mario.GetDistances();
        float[] marioPosition = {marioVector.x, marioVector.y};
        float marioRotation = mario.transform.eulerAngles.y;
        float[] coinPosition = {coinVector.x, coinVector.y};

        Observation obs = new Observation(distance, marioDistances, 
                                            marioPosition, marioRotation, 
                                            coinPosition, smallCoinsCollectedCount);

        Pipeline.WriteObservation(obs);
    }

    #endregion

}

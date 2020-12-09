using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class Environment : MonoBehaviour
{

    public int id = 0;
    public Mario mario;
    public SmallCoin smallCoin;

    public float minX = -4;
    public float maxX = 4;
    public float minZ = -4;
    public float maxZ = 4;

    const int maxSmallCoinCount = 10;
    const float smallCoinFixedY = 1.25f;
    int smallCoinsCollectedCount = 0;

    float[] marioYPositions = {0};
    public Transform[] coinPositions;
    int evalCoinPositions = 10;

    public GameObject obstacle;

    #region Events

    public void CollectedSmallCoin()
    {
        smallCoinsCollectedCount += 1;
    }

    #endregion

    #region Setup

    Vector3 CreateRandomPosition()
    {
        return new Vector3(Random.Range(minX, maxX), 0, Random.Range(minZ, maxZ));
    }

    void AddCoin(Vector3 position)
    {
        SmallCoin sc = SmallCoin.Instantiate(smallCoin);
        sc.gameObject.SetActive(true);
        sc.environment = this;
        sc.transform.position = position;
        sc.transform.parent = transform;
    }

    void Setup()
    {
        SetupWall();

        Vector3 randomPosition = CreateRandomPosition();
        mario.SetPosition(randomPosition + transform.position);

        int scc = 0;
        while (scc < maxSmallCoinCount)
        {
            randomPosition = CreateRandomPosition();
            randomPosition.y = smallCoinFixedY;
            float distance = Vector3.Distance(mario.transform.position, randomPosition);
            if (distance < 3)
            {
                continue;
            }

            AddCoin(randomPosition + transform.position);
            scc += 1;
        }
    }

    void SetupEval()
    {
        SetupWall();

        Vector3 randomPosition = CreateRandomPosition();
        randomPosition.z = marioYPositions[id % marioYPositions.Length];
        randomPosition.x = minX;
        mario.SetPosition(randomPosition + transform.position);

        int startIndex = (id * 10) % 50;
        int endIndex = startIndex + 10; 
        for (int i = startIndex; i < endIndex ; i++)
        {
            AddCoin(coinPositions[i].position);
        }
    }

    void SetupWall()
    {
        obstacle.transform.eulerAngles = new Vector3(
            obstacle.transform.eulerAngles.x,
            Random.Range(0f, 360f),
            obstacle.transform.eulerAngles.z
        );

        obstacle.transform.localScale = new Vector3(
            Random.Range(0.4f, 0.75f),
            obstacle.transform.localScale.y,
            obstacle.transform.localScale.z
        );
    }

    public void Reset()
    {
        smallCoinsCollectedCount = 0;
        Setup();
        
        Pipeline.WriteGameStarted();
    }

    public void ResetEval()
    {
        smallCoinsCollectedCount = 0;
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

    public void SetAgentAction(Action action)
    {
        mario.currentAction = action;
    }

    void WriteObservation()
    {
        Observation obs = GetObservation();

        Pipeline.WriteObservation(obs);
    }

    public Observation GetObservation()
    {
        Vector2 marioVector = new Vector2(mario.transform.position.x - transform.position.x, 
                                          mario.transform.position.z - transform.position.z);
        float distance = mario.GetNearestCoinDistance();
    
        float[] marioDistances = mario.GetDistances();
        float[] marioPosition = {marioVector.x, marioVector.y};
        float marioRotation = mario.transform.eulerAngles.y;

        Observation obs = new Observation(distance, marioDistances, marioPosition, 
                                          marioRotation, smallCoinsCollectedCount);
        return obs;
    }

    #endregion

}

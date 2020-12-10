using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnvironmentManager : MonoBehaviour
{
    public Environment environmentObject;

    public Environment[] environments;
    const float environmentSpacing = 12;

    float updateFrequency = 0.1f;
    float lastUpdateTime = 0;

    public delegate void ResetEvent();
    public static event ResetEvent ResetState;

    public static bool drawCoinVision = true;
    public static bool drawWallVision = true;

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
            Reset();
            return;
        }
        if (Input.GetKeyUp(KeyCode.Alpha1))
        {
            EnvironmentManager.drawCoinVision = !EnvironmentManager.drawCoinVision;
        }
        if (Input.GetKeyUp(KeyCode.Alpha2))
        {
            EnvironmentManager.drawWallVision = !EnvironmentManager.drawWallVision;
        }

        if  (Time.time - lastUpdateTime > updateFrequency) 
        {
            SetAgentActions();
            HandleGameState();
            WriteObservations();
            
            lastUpdateTime = Time.time;
        }
    }

    #region Lifecycle

    void AddEnvironment(int index)
    {
        Environment env = Environment.Instantiate(environmentObject);
        env.id = index;
        env.gameObject.SetActive(true);
        float y = 0;
        if (index > 2) {
            y -= environmentSpacing;
            if (index > 5) {
                y -= environmentSpacing;
            }
        }

        env.transform.position = new Vector3((index % 3) * environmentSpacing, 0, y);
        env.transform.parent = transform;
        environments[index] = env;
    }

    void Reset()
    {
        if (ResetState != null)
        {
            ResetState();
        }
        Pipeline.ClearAction(environments.Length);
        for (int i = 0; i < environments.Length; i++)
        {
            environments[i].Reset();
        }
    }

    void ResetEval()
    {
        if (ResetState != null)
        {
            ResetState();
        }
        Pipeline.ClearAction(environments.Length);
        for (int i = 0; i < environments.Length; i++)
        {
            environments[i].ResetEval();
        }
    }

    #endregion

    #region I/O

    void HandleGameState()
    {
        int isGameOver = Pipeline.ReadIsGameOver();
        if (isGameOver == GameState.isGameover)
        {
            Reset();
        } 
        else if (isGameOver == GameState.isEvalGameover)
        {
            ResetEval();
        }
    }

    void SetAgentActions()
    {
        for (int i = 0; i < environments.Length; i++)
        {
            Environment env = environments[i];
            Action action = Pipeline.ReadAction(i);
            env.SetAgentAction(action);
        }
    }

    void WriteObservations()
    {
        Observation[] observations = new Observation[environments.Length];
        for (int i = 0; i < environments.Length; i++)
        {
            Observation obs = environments[i].GetObservation();
            observations[i] = obs;
        }

        Pipeline.WriteObservations(observations);
    }

    #endregion
}

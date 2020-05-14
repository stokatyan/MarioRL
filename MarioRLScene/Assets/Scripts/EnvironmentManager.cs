using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnvironmentManager : MonoBehaviour
{
    public Environment environmentObject;

    public Environment[] environments;
    const float environmentSpacing = 14;

    float updateFrequency = 0.1f;
    float lastUpdateTime = 0;

    public delegate void ResetEvent();
    public static event ResetEvent ResetState;

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
        env.transform.position = new Vector3(index * environmentSpacing, 0, 0);
        env.transform.parent = transform;
        environments[index] = env;
    }

    void Reset()
    {
        // Pipeline.ClearAction();
        for (int i = 0; i < environments.Length; i++)
        {
            environments[i].Reset();
        }
    }

    void ResetEval()
    {
        // Pipeline.ClearAction();
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
        Action[] actions = Pipeline.ReadActions();
        for (int i = 0; i < environments.Length; i++)
        {
            Environment env = environments[i];
            if (i < actions.Length)
            {
                env.SetAgentAction(actions[i]);
            }
            else 
            {
                Action noAction = new Action();
                env.SetAgentAction(noAction);
            }
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

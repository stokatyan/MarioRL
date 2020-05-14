﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;


public class Pipeline
{
    static string observationPath = "Assets/Resources/environment_output.txt";
    static string gamestatePath = "Assets/Resources/environment_state.txt";
    static string actionPath = "Assets/Resources/environment_input_";
    static string actionPathSuffix = ".txt";

    public static Action ReadAction(int index)
    {
        string path = actionPath + index.ToString() + actionPathSuffix;
        StreamReader reader = new StreamReader(path);
        string json = reader.ReadToEnd();
        reader.Close();

        if (json.Length < 5)
        {
            return new Action();
        }
        Action obj = JsonUtility.FromJson<Action>(json);
        return obj;
    }

    public static int ReadIsGameOver()
    {
        StreamReader reader = new StreamReader(gamestatePath);
        string json = reader.ReadToEnd();
        reader.Close();

        if (json.Length < 5)
        {
            return 0;
        }
        GameState obj = JsonUtility.FromJson<GameState>(json);
        return obj.gameover;
    }

    public static void ClearAction(int numActionPaths)
    {
        for (int i = 0; i < numActionPaths; i++)
        {
            string path = actionPath + i.ToString() + actionPathSuffix;
            StreamWriter writer = new StreamWriter(path, false);
            writer.WriteLine("");
            writer.Close();
        }
        
    }

    public static void WriteObservation(Observation obs)
    {
        string json = JsonUtility.ToJson(obs);
        StreamWriter writer = new StreamWriter(observationPath, false);
        writer.WriteLine(json);
        writer.Close();
    }

    public static void WriteObservations(Observation[] observations)
    {
        string json = JsonHelper.ToJson(observations);
        StreamWriter writer = new StreamWriter(observationPath, false);
        writer.WriteLine(json);
        writer.Close();
    }

    public static void WriteGameStarted()
    {
        GameState state = new GameState();
        string json = JsonUtility.ToJson(state);
        StreamWriter writer = new StreamWriter(gamestatePath, false);
        writer.WriteLine(json);
        writer.Close();
    }
    
}


[System.Serializable]
public class GameState
{
    public const int isNotGameover = 0;
    public const int isGameover = 1;
    public const int isEvalGameover = 2;

    [SerializeField]
    public int gameover = 0;
}

[System.Serializable]
public class Observation
{
    [SerializeField]
    public float distance = 0;

    [SerializeField]
    public float[] smallCoinDistances = new float[19];

    [SerializeField]
    public float[] marioPosition = new float[2];

    [SerializeField]
    public float marioRotation = 0;

    [SerializeField]
    public int smallCoinsCollected = 0;

    const float maxRotation = 360;

    public Observation(float distance, float[] smallCoinDistances, float[] marioPosition, 
                        float marioRotation, int smallCoinsCollected)
    {
        this.distance = distance;
        this.smallCoinDistances = smallCoinDistances;
        this.marioPosition = marioPosition;
        this.marioRotation = marioRotation;

        if (marioRotation < 0)
        {
            this.marioRotation += maxRotation;
        }
        this.marioRotation /= maxRotation;

        this.smallCoinsCollected = smallCoinsCollected;
    }
}

[System.Serializable]
public class Action
{
    [SerializeField]
    public float left = 0;
    [SerializeField]
    public float right = 0;
    [SerializeField]
    public float up = 0;
    [SerializeField]
    public float down = 0;
}
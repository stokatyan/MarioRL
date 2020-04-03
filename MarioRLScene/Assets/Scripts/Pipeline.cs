using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;


public class Pipeline
{

    static string observationPath = "Assets/Resources/environment_output.txt";
    static string gamestatePath = "Assets/Resources/environment_state.txt";
    static string actionPath = "Assets/Resources/environment_input.txt";

    public static Action ReadAction()
    {
        StreamReader reader = new StreamReader(actionPath);
        string json = reader.ReadToEnd();
        reader.Close();

        if (json.Length < 5)
        {
            return new Action();
        }
        Action obj = JsonUtility.FromJson<Action>(json);
        return obj;
    }

    public static bool ReadIsGameOver()
    {
        StreamReader reader = new StreamReader(gamestatePath);
        string json = reader.ReadToEnd();
        reader.Close();

        if (json.Length < 5)
        {
            return false;
        }
        GameState obj = JsonUtility.FromJson<GameState>(json);
        return obj.isGameOver;
    }

    public static void ClearAction()
    {
        StreamWriter writer = new StreamWriter(actionPath, false);
        writer.WriteLine("");
        writer.Close();
    }

    public static void WriteObservation(Observation obs)
    {
        string json = JsonUtility.ToJson(obs);
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
    [SerializeField]
    public int gameover = 0;

    public bool isGameOver
    {
        get { return gameover == 1; }
    }

}

[System.Serializable]
public class Observation
{
    [SerializeField]
    public float distance = 0;

    [SerializeField]
    public float[] marioPosition = new float[2];

    [SerializeField]
    public float[] coinPosition = new float[2];

    [SerializeField]
    public int smallCoinsCollectedCount = 0;

    public Observation(float distance, float[] marioPosition, float[] coinPosition, int smallCoinsCollectedCount)
    {
        this.distance = distance;
        this.marioPosition = marioPosition;
        this.coinPosition = coinPosition;
        this.smallCoinsCollectedCount = smallCoinsCollectedCount;
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
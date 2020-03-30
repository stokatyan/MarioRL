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

    public static void WriteGameOver()
    {
        StreamWriter writer = new StreamWriter(gamestatePath, false);
        writer.WriteLine("{gameover:1}");
        writer.Close();
    }

    public static void WriteGameStarted()
    {
        StreamWriter writer = new StreamWriter(gamestatePath, false);
        writer.WriteLine("{gameover:0}");
        writer.Close();
    }
    
}


[System.Serializable]
public class Observation
{
    [SerializeField]
    public float distance = 0;

    public Observation(float distance)
    {
        this.distance = distance;
    }
}

[System.Serializable]
public class Action
{
    [SerializeField]
    public bool left = false;
    [SerializeField]
    public bool right = false;
    [SerializeField]
    public bool up = false;
    [SerializeField]
    public bool down = false;
}
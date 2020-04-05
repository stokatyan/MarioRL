using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Tags : MonoBehaviour
{
    public const string coin = "coin";
    public const string mario = "mario";
    public const string smallCoin = "smallCoin";
    public const string wall = "wall";
    
}

public class Layers : MonoBehaviour
{
    public const int SmallCoinIndex = 8;
    public const int WallIndex = 9;
    public const int SmallCoinMask = 1 << SmallCoinIndex;
    public const int WallMask = 1 << WallIndex;
    public const int SmallCoinAndWallMask = SmallCoinMask | WallMask;
}

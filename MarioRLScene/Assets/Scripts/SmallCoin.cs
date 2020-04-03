using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SmallCoin : Coin
{

    public delegate void CollectedEvent();
    public static event CollectedEvent Collected;

    // Start is called before the first frame update
    void Start()
    {
        shouldAnimateBounce = false;
    }

    protected new void Update()
    {
        base.Update();
    }

    override protected void Reset()
    {
        Destroy(this.gameObject);
    }

    protected new void OnTriggerEnter(Collider other) {
        if (Collected != null)
        {
            Collected();
        }
        Destroy(this.gameObject);
     }

}

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SmallCoin : Coin
{

    [HideInInspector]
    public Environment environment;

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
        environment.CollectedSmallCoin();
        Destroy(this.gameObject);
     }

}

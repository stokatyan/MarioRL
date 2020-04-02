using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Mario : MonoBehaviour
{

    public float movementForce = 5f;
    public float rotationSpeed;

    Rigidbody rb;
    Animator animator;
    
    [HideInInspector]
    public Action currentAction;

    void Awake()
    {
        animator = GetComponent<Animator>();
        rb = GetComponent<Rigidbody>();
        
    }

    void FixedUpdate()
    {
        HandleMovement();
    }

    #region Movement

    void HandleMovement()
    {
        Vector3 movementVector = Vector3.zero;
        bool shouldMove = true;

        movementVector += Vector3.forward * currentAction.up;
        movementVector -= Vector3.forward * currentAction.down;
        movementVector += Vector3.right * currentAction.right;
        movementVector -= Vector3.right * currentAction.left;

        if (Input.GetKey(KeyCode.UpArrow))
        {
            movementVector += Vector3.forward;
        }
        if (Input.GetKey(KeyCode.DownArrow))
        {
            movementVector -= Vector3.forward;
        }
        if (Input.GetKey(KeyCode.RightArrow))
        {
            movementVector += Vector3.right;
        }
        if (Input.GetKey(KeyCode.LeftArrow))
        {
            movementVector -= Vector3.right;
        }
        if (movementVector == Vector3.zero)
        {
            shouldMove = false;
        }

        animator.SetBool(AnimationKeys.isMoving, shouldMove);
        if (shouldMove)
        {
            rb.AddForce(movementVector * movementForce);
            RotateTowardsVector(movementVector);
        } else 
        {
            rb.velocity = Vector3.zero;
        }
                
    }

    void RotateTowardsVector(Vector3 vector)
    {
        Quaternion lookRotation = Quaternion.LookRotation(vector); 
        rb.MoveRotation(Quaternion.Lerp(transform.rotation, lookRotation, Time.deltaTime * rotationSpeed));
    }

    public void SetPosition(Vector3 position)
    {
        rb.MovePosition(position);
    }

    #endregion

    #region Collision

    void OnTriggerEnter(Collider other) {
         if (other.tag == Tags.coin) {
            //  Destroy(other.gameObject);
         }
     }

    #endregion

    #region State

    void OnEnable()
    {
        Environment.ResetState += Reset;
    }


    void OnDisable()
    {
        Environment.ResetState -= Reset;
    }


    void Reset()
    {
        RotateTowardsVector(Vector3.forward);
        currentAction = new Action();
    }

    #endregion


}

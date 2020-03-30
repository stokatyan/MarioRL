using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Mario : MonoBehaviour
{

    public float movementForce = 5f;
    public float rotationSpeed;

    Rigidbody rb;
    Animator animator;

    Vector3 targetRotation = Vector3.zero;
    
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
        bool shouldMove = false;
        if (Input.GetKey(KeyCode.UpArrow) || currentAction.up)
        {
            movementVector += Vector3.forward;
            shouldMove = true;
        }
        if (Input.GetKey(KeyCode.DownArrow) || currentAction.down)
        {
            movementVector -= Vector3.forward;
            shouldMove = true;
        }
        if (Input.GetKey(KeyCode.RightArrow) || currentAction.right)
        {
            movementVector += Vector3.right;
            shouldMove = true;
        }
        if (Input.GetKey(KeyCode.LeftArrow) || currentAction.left)
        {
            movementVector -= Vector3.right;
            shouldMove = true;
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
        if (vector == targetRotation)
        {
            return;
        }

        targetRotation = vector;
        Quaternion lookRotation = Quaternion.LookRotation(vector); 
        StartCoroutine(RotateTo(lookRotation));
    }

    IEnumerator RotateTo(Quaternion target) {
        Quaternion from = transform.rotation;
        for ( float t = 0f; t < 1f; t += rotationSpeed*Time.deltaTime) {
            transform.rotation = Quaternion.Lerp(from, target, t);
            yield return null;
        }
        transform.rotation = target;
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

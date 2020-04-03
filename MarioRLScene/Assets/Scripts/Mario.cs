﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Mario : MonoBehaviour
{

    public float movementForce;
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

        RaycastSight();
        
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

        if (Input.GetKey(KeyCode.LeftShift))
        {
            movementVector /= 2;
        }

        animator.SetFloat(AnimationKeys.moveSpeed, movementVector.magnitude);

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

    #region Visio

    void RaycastSight()
    {        
        Vector3 fwd = transform.TransformDirection(Vector3.forward);
        Vector3 lft = transform.TransformDirection(Vector3.left);
        Vector3 lft_1 = transform.TransformDirection(Vector3.left + Vector3.forward/3);
        RaycastToDirection(fwd);
        RaycastToDirection(lft);
        RaycastToDirection(lft_1);
    }

    RaycastHit RaycastToDirection(Vector3 direction)
    {
        Vector3 rayStart = transform.position + direction * 0.5f;
        rayStart.y = 1.25f;

        RaycastHit hit;
        if (Physics.Raycast(rayStart, direction, out hit, Mathf.Infinity, Layers.SmallCoinMask))
        {
            Debug.DrawRay(rayStart, direction * hit.distance, Color.green);
        } else {
            Debug.DrawRay(rayStart, direction * 1000, Color.red);
        }

        return hit;
    }

    #endregion

}

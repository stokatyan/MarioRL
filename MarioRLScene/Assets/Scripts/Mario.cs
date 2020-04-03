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
        Vector3 rht = transform.TransformDirection(Vector3.right);
        for (float i = 1; i <= 10; i++)
        {
            Vector3 direction = lft*((10-i)/10) + fwd*((i + i)/10.0f);
            RaycastToDirection(direction, fwd);
        }
        for (float i = 1; i < 10 ; i++)
        {
            Vector3 direction = rht*((10-i)/10) + fwd*((i + i)/10.0f);
            RaycastToDirection(direction, fwd, true);
        }
    }

    RaycastHit RaycastToDirection(Vector3 direction, Vector3 fwd, bool isRightSide = false)
    {
        float rootAngle = Vector3.SignedAngle(fwd, Vector3.forward, Vector3.up);
        if (isRightSide)
        {
            rootAngle *= -1;
        }

        float angle = (rootAngle + Vector3.Angle(direction, fwd)) * 0.0174533f;
        float radius = 1;
        float x = radius * Mathf.Sin(angle);
        if (!isRightSide)
        {
            x *= -1;
        }

        float y = 1.25f;
        float z = radius * Mathf.Cos(angle);
        Vector3 rayStart = transform.position;
        rayStart.y = y;
        Vector3 rayStart2 = rayStart + new Vector3(x, 0, z);
        RaycastHit hit;

        if (Physics.Raycast(rayStart, direction, out hit, Mathf.Infinity, Layers.SmallCoinMask))
        {
            Debug.DrawRay(rayStart2, direction * hit.distance, Color.green);
        } else {
            Debug.DrawRay(rayStart2, direction * 15, Color.red);
        }

        return hit;
    }

    #endregion

}

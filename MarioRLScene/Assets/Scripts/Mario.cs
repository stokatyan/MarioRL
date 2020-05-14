using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Mario : MonoBehaviour
{

    const float maxCoinDistance = 12;

    public float movementForce;
    public float rotationSpeed;

    Rigidbody rb;
    Animator animator;
    
    [HideInInspector]
    public Action currentAction;

    float[] distances = new float[19];

    Vector3 nearestCoinPosition;
    float nearestCoinDistance = maxCoinDistance;

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

    void OnTriggerEnter(Collider other) {
        if (other.gameObject.tag == Tags.smallCoin) 
        {
            nearestCoinDistance = maxCoinDistance;
        }
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

    public float[] GetDistances() 
    {
        return distances;
    }

    public float GetNearestCoinDistance()
    {
        return nearestCoinDistance;
    }

    void OnEnable()
    {
        EnvironmentManager.ResetState += Reset;
    }


    void OnDisable()
    {
        EnvironmentManager.ResetState -= Reset;
    }


    void Reset()
    {
        nearestCoinDistance = maxCoinDistance;
        RotateTowardsVector(Vector3.forward);
        currentAction = new Action();
    }

    #endregion

    #region Vision

    void RaycastSight()
    {       

        if (nearestCoinDistance < maxCoinDistance)
        {
            nearestCoinDistance = Vector3.Distance(nearestCoinPosition, transform.position);
        }

        Vector3 fwd = transform.TransformDirection(Vector3.forward);
        Vector3 lft = transform.TransformDirection(Vector3.left);
        Vector3 rht = transform.TransformDirection(Vector3.right);

        int distanceIndex = 0;

        for (float i = 1; i <= 10; i++)
        {
            Vector3 direction = lft*((10-i)/10) + fwd*((i + i)/10.0f);
            bool isForward = i == 10;
            float d = RaycastToDirection(direction, fwd, false, isForward);
            distances[distanceIndex] = d;

            distanceIndex += 1;
        }
        for (float i = 1; i < 10; i++)
        {
            Vector3 direction = rht*((i)/10) + fwd*((10-i) * 2/10.0f);
            float d = RaycastToDirection(direction, fwd, true);
            distances[distanceIndex] = d;

            distanceIndex += 1;
        }
    }

    float RaycastToDirection(Vector3 direction, Vector3 fwd, bool isRightSide = false, bool isForward = false)
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

        float hitDistance = 12;
        if (Physics.Raycast(rayStart, direction, out hit, Mathf.Infinity, Layers.SmallCoinAndWallMask))
        {
            Color hitColor = Color.green;
            Color missColor = Color.red;
            if (isForward)
            {
                hitColor = Color.yellow;
                missColor = Color.magenta;
            }
            if (hit.transform.gameObject.tag == Tags.smallCoin)
            {
                Debug.DrawRay(rayStart2, hit.point - rayStart2, hitColor);
                hitDistance = hit.distance;

                UpdateNearestCoinDistance(hit.transform.position);
                
            } else {
                Debug.DrawRay(rayStart2, hit.point - rayStart2, missColor);
            }            
        }

        return hitDistance;
    }

    void UpdateNearestCoinDistance(Vector3 newCoinPosition)
    {
        Vector2 v1 = new Vector2(transform.position.x, transform.position.z);
        Vector2 v2 = new Vector2(newCoinPosition.x, newCoinPosition.z);

        float coinDistance = Vector3.Distance(v1, v2);
        if (coinDistance < nearestCoinDistance)
        {
            nearestCoinDistance = coinDistance;
            nearestCoinPosition = newCoinPosition;
        }
    }

    #endregion

}

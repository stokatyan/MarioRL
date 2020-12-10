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

    float[] distances = new float[26];

    float visionLineWidth = 0.04f;

    LineRenderer[] lines = new LineRenderer[26]; // 19 coin detectors, 7 wall detectors
    public Material hitMat, missMat, wallDetectionMat;

    Vector3 nearestCoinPosition;
    float nearestCoinDistance = maxCoinDistance;

    void Awake()
    {
        animator = GetComponent<Animator>();
        rb = GetComponent<Rigidbody>();
    }

    void Start()
    {
        SetupLines();
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

    #region Setup

    void SetupLines()
    {
        for (int i = 0; i < lines.Length; i++)
        {
            var go = new GameObject();
            var lr = go.AddComponent<LineRenderer>();
            lr.startWidth = visionLineWidth;
            lr.endWidth = visionLineWidth;
            lr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            lr.receiveShadows = false;
            lines[i] = lr;
            go.transform.parent = gameObject.transform;
        }
    }

    #endregion

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

        float coinVisionHalfCount = 10;
        for (float i = 1; i <= coinVisionHalfCount; i++)
        {
            Vector3 direction = lft*((coinVisionHalfCount-i)/coinVisionHalfCount) + fwd*((i + i)/coinVisionHalfCount);
            float d = RaycastToDirection(direction, fwd, distanceIndex, false);
            distances[distanceIndex] = d;

            distanceIndex += 1;
        }
        for (float i = 1; i < coinVisionHalfCount; i++)
        {
            Vector3 direction = rht*((i)/coinVisionHalfCount) + fwd*((coinVisionHalfCount-i) * 2/coinVisionHalfCount);
            float d = RaycastToDirection(direction, fwd, distanceIndex, true);
            distances[distanceIndex] = d;

            distanceIndex += 1;
        }


        float wallVisionHalfCount = 4;
        for (float i = 1; i <= wallVisionHalfCount; i++)
        {
            Vector3 direction = lft*((wallVisionHalfCount-i)/wallVisionHalfCount) + fwd*((i + i)/wallVisionHalfCount);
            float d = RaycastToDirection(direction, fwd, distanceIndex, false, true);
            distances[distanceIndex] = d;

            distanceIndex += 1;
        }
        for (float i = 1; i < wallVisionHalfCount; i++)
        {
            Vector3 direction = rht*((i)/wallVisionHalfCount) + fwd*((wallVisionHalfCount-i) * 2/wallVisionHalfCount);
            float d = RaycastToDirection(direction, fwd, distanceIndex, true, true);
            distances[distanceIndex] = d;

            distanceIndex += 1;
        }
    }

    float RaycastToDirection(Vector3 direction, Vector3 fwd, int index, bool isRightSide, bool isWallVision = false)
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
        if (isWallVision) {
            y -= 0.1f;
            lines[index].enabled = EnvironmentManager.drawWallVision;
        } else {
            lines[index].enabled = EnvironmentManager.drawCoinVision;
        }
        float z = radius * Mathf.Cos(angle);
        Vector3 rayStart = transform.position;
        rayStart.y = y;
        Vector3 rayStart2 = rayStart + new Vector3(x, 0, z);
        RaycastHit hit;
        float startOffset = Mathf.Abs(Vector3.Distance(rayStart, rayStart2));

        lines[index].material = wallDetectionMat;
        lines[index].SetPosition(0, rayStart2);  

        float hitDistance = maxCoinDistance;
        float endpointDistance = hitDistance;
        Vector3 endpoint;

        int mask = Layers.SmallCoinAndWallMask;
        if (isWallVision) {
            mask = Layers.WallMask;
        }

        if (Physics.Raycast(rayStart, direction, out hit, Mathf.Infinity, mask))
        {
            endpointDistance = hit.distance;

            if (hit.transform.gameObject.tag == Tags.smallCoin && !isWallVision)
            {
                lines[index].material = hitMat;
                hitDistance = hit.distance;
                UpdateNearestCoinDistance(hit.transform.position);
            } else if (isWallVision) {
                hitDistance = hit.distance;
            } else {
                lines[index].material = missMat;
            }
        } else {
            
            if (!isWallVision) {
                lines[index].material = missMat;
            }
        }

        Ray ray = new Ray(rayStart2, direction);
        endpoint = ray.GetPoint(endpointDistance - startOffset);
        lines[index].SetPosition(1, endpoint);

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

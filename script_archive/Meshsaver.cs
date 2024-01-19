using Microsoft.MixedReality.SceneUnderstanding;
using System.Collections;
using System.Linq;
using System.IO;
using UnityEngine;

public class MeshSaver : MonoBehaviour
{
    public float saveInterval = 20.0f;  // Updated save interval to 60 seconds
    private int fileCounter = 0;
    private const int maxRetries = 3;  // Define a maximum number of retries
    private const float retryDelay = 5.0f;  // Delay in seconds before retrying

    public GameObject cube; // Drag the cube GameObject here in the Inspector

    private void Start()
    {
        StartCoroutine(SaveMeshRoutine());
    }

    private void SetCubeColorBlack()
    {
        // Set the cube color to black: Init state
        if (cube)
        {
            cube.GetComponent<Renderer>().material.color = Color.black;
        }
    }

    private void SetCubeColorSuccess()
    {
        if (cube)
        {
            cube.GetComponent<Renderer>().material.color = Color.green;
        }
    }

    private void SetCubeColorFail()
    {
        if (cube)
        {
            cube.GetComponent<Renderer>().material.color = Color.red;
        }
    }

    private void SetCubeColorInProgress()
    {
        if (cube)
        {
            cube.GetComponent<Renderer>().material.color = Color.yellow;
        }
    }

    private void SetCubeColorFailIO()
    {
        if (cube)
        {
            cube.GetComponent<Renderer>().material.color = Color.blue;
        }
    }

    private void SetCubeColorFailMemory()
    {
        if (cube)
        {
            cube.GetComponent<Renderer>().material.color = Color.white;
        }
    }


    private IEnumerator SaveMeshRoutine()
    {
        SetCubeColorBlack();
        while (true)
        {
            yield return new WaitForSeconds(saveInterval);
            SaveSceneMeshes();
        }
    }

    private async void SaveSceneMeshes(int retryCount = 0)
    {
        SetCubeColorInProgress(); // Set color to yellow at the beginning, indicating "in progress"
        Debug.Log("-------- Starting save scene meshes --------");
        try
        {
            Debug.Log("_____0_____");

            float searchRadius = 5.0f;
            var querySettings = new SceneQuerySettings()
            {
                EnableSceneObjectMeshes = true,
                EnableSceneObjectQuads = false,
                EnableWorldMesh = false
            };

            Debug.Log("_____1_____");

            Scene scene = null;

            if (!SceneObserver.IsSupported())
            {
                Debug.LogError("SceneUnderstandingManager.Start: Scene Understanding not supported.");
                return;
            }

            Debug.Log("_____1.1_____");

            SceneObserverAccessStatus access = await SceneObserver.RequestAccessAsync();
            if (access != SceneObserverAccessStatus.Allowed)
            {
                Debug.LogError("SceneUnderstandingManager.Start: Access to Scene Understanding has been denied.\n" +
                                "Reason: " + access);
                return;
            }

            Debug.Log("_____1.2_____");

            scene = await SceneObserver.ComputeAsync(querySettings, searchRadius);

            Debug.Log("_____2_____");

            if (scene == null) throw new System.Exception("Scene is null.");
            Debug.Log("_____3_____");


            foreach (var sceneObject in scene.SceneObjects)
            {
                Debug.Log("_____4_____");

                if (sceneObject == null) continue;
                Debug.Log("_____5_____");

                string label = sceneObject.Kind.ToString();

                Debug.Log("_____6_____");


                // If there are multiple meshes, select the largest one based on triangle count
                SceneMesh largestMesh = sceneObject.Meshes.OrderByDescending(m => m.TriangleIndexCount).FirstOrDefault();

                Debug.Log("_____7_____");

                if (largestMesh != null)
                {
                    Debug.Log("_____8_____");

                    Mesh unityMesh = ConvertSceneMeshToUnityMesh(largestMesh);

                    Debug.Log("_____9_____");

                    SaveMeshToFile(unityMesh, $"{Application.persistentDataPath}/{label}_{fileCounter:0000}.dat");
                    Debug.Log("_____26_____");
                }
            }

            fileCounter = (fileCounter + 1) % 10000;
            SetCubeColorSuccess();
        }
        catch (System.Exception e)
        {
            if (e is System.IO.IOException) {
                Debug.LogError($"IO Error: {e.Message}");
                Debug.Log($"IO Error: {e.Message}");
                SetCubeColorFailIO();
            } else if (e is System.OutOfMemoryException) {
                Debug.LogError($"Memory Error: {e.Message}");
                Debug.Log($"Memory Error: {e.Message}");
                SetCubeColorFailMemory();
            } else {
                Debug.LogError($"Error in SaveSceneMeshes: {e.Message}");
                Debug.Log($"Error in SaveSceneMeshes: {e.Message}");
                SetCubeColorFail();
            }
            
            
            // Backup logic to retry
            if (retryCount < maxRetries)
            {
                Debug.Log($"Retrying in {retryDelay} seconds. Retry {retryCount + 1}/{maxRetries}");
                StartCoroutine(RetrySaveMeshes(retryCount + 1));
            }
            else
            {
                Debug.LogError("Max retries reached. Giving up this save attempt.");
            }
        }
    }

    private IEnumerator RetrySaveMeshes(int retryCount)
    {
        yield return new WaitForSeconds(retryDelay);
        SaveSceneMeshes(retryCount);
    }

    private Mesh ConvertSceneMeshToUnityMesh(SceneMesh sceneMesh)
    {
        Mesh unityMesh = new Mesh();

        // Use the correct methods to get vertices and indices
        System.Numerics.Vector3[] positions = new System.Numerics.Vector3[sceneMesh.VertexCount];
        uint[] indices = new uint[sceneMesh.TriangleIndexCount];

        sceneMesh.GetVertexPositions(positions);
        sceneMesh.GetTriangleIndices(indices);

        // Convert System.Numerics.Vector3 to UnityEngine.Vector3
        Vector3[] unityVertices = positions.Select(p => new Vector3(p.X, p.Y, p.Z)).ToArray();

        int[] unityIndices = indices.Select(i => (int)i).ToArray();

        unityMesh.vertices = unityVertices;
        unityMesh.triangles = unityIndices;

        return unityMesh;
    }

    public void SaveMeshToFile(Mesh mesh, string filename)
    {
        Debug.Log("_____10_____");
        if (mesh == null) return;
        Debug.Log("_____11_____");
        byte[] meshData = SerializeMesh(mesh);
        Debug.Log("_____24_____");
        File.WriteAllBytes(filename, meshData);
        Debug.Log("_____25_____");
    }

    public byte[] SerializeMesh(Mesh mesh)
    {
        Debug.Log("_____12_____");
        using (MemoryStream stream = new MemoryStream())
        {
            Debug.Log("_____13_____");
            BinaryWriter writer = new BinaryWriter(stream);
            Debug.Log("_____14_____");
            Vector3[] vertices = mesh.vertices;
            Debug.Log("_____15_____");
            writer.Write(vertices.Length);
            Debug.Log("_____16_____");
            foreach (var vertex in vertices)
            {
                Debug.Log("_____17_____");
                writer.Write(vertex.x);
                writer.Write(vertex.y);
                writer.Write(vertex.z);
                Debug.Log("_____18_____");
            }
            int[] triangles = mesh.triangles;
            Debug.Log("_____19_____");
            writer.Write(triangles.Length);
            Debug.Log("_____20_____");
            foreach (var triangle in triangles)
            {
                Debug.Log("_____21_____");
                writer.Write(triangle);
                Debug.Log("_____22_____");
            }
            Debug.Log("_____23_____");
            return stream.ToArray();
        }
    }
}

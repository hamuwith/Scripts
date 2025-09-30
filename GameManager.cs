using UnityEngine;
using UnityEngine.SceneManagement;

public class GameManager : MonoBehaviour
{
    /// <summary> シングルトンインスタンス </summary>
    public static GameManager Instance { get; private set; }
    /// <summary> シーンの状態を管理する変数 </summary>
    [SerializeField] SceneState currectScene;
    /// <summary> シーンの状態を管理する列挙型 </summary>
    enum SceneState
    {
        Title,
        Main,
        Story,
    }
    private void Awake()
    {
        //インスタンスが存在しない場合は、現在のインスタンスを設定し、破棄しないようにする
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
        }
    }
    private void Start()
    {
        //シーンがロードされたときのイベントを登録
        SceneManager.sceneLoaded += OnSceneLoaded;
    }
    /// <summary>
    /// シーンの遷移を行う
    /// </summary>
    /// <param name="sceneState">シーン状態</param>
    private void LoadScene(SceneState sceneState)
    {
        //シーンをIDで読み込み
        SceneManager.LoadScene((int)sceneState);
    }
    /// <summary>
    /// シーンがロードされたときの処理
    /// </summary>
    /// <param name="scene">シーン</param>
    /// <param name="mode">モード</param>
    private void OnSceneLoaded(Scene scene, LoadSceneMode mode)
    {
        currectScene = (SceneState)scene.buildIndex; // 現在のシーンの状態を更新

        //シーンの状態に応じた処理を行う
        switch (scene.buildIndex)
        {
            case 0: // タイトルシーン
                Debug.Log("タイトルシーンがロードされました");
                break;
            case 1: // メインシーン
                Debug.Log("メインシーンがロードされました");
                break;
            case 2: // ストーリーシーン
                Debug.Log("ストーリーシーンがロードされました");
                break;
        }
    }
}

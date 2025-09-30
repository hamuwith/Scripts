using UnityEngine;
using UnityEngine.SceneManagement;

public class GameManager : MonoBehaviour
{
    /// <summary> �V���O���g���C���X�^���X </summary>
    public static GameManager Instance { get; private set; }
    /// <summary> �V�[���̏�Ԃ��Ǘ�����ϐ� </summary>
    [SerializeField] SceneState currectScene;
    /// <summary> �V�[���̏�Ԃ��Ǘ�����񋓌^ </summary>
    enum SceneState
    {
        Title,
        Main,
        Story,
    }
    private void Awake()
    {
        //�C���X�^���X�����݂��Ȃ��ꍇ�́A���݂̃C���X�^���X��ݒ肵�A�j�����Ȃ��悤�ɂ���
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
        //�V�[�������[�h���ꂽ�Ƃ��̃C�x���g��o�^
        SceneManager.sceneLoaded += OnSceneLoaded;
    }
    /// <summary>
    /// �V�[���̑J�ڂ��s��
    /// </summary>
    /// <param name="sceneState">�V�[�����</param>
    private void LoadScene(SceneState sceneState)
    {
        //�V�[����ID�œǂݍ���
        SceneManager.LoadScene((int)sceneState);
    }
    /// <summary>
    /// �V�[�������[�h���ꂽ�Ƃ��̏���
    /// </summary>
    /// <param name="scene">�V�[��</param>
    /// <param name="mode">���[�h</param>
    private void OnSceneLoaded(Scene scene, LoadSceneMode mode)
    {
        currectScene = (SceneState)scene.buildIndex; // ���݂̃V�[���̏�Ԃ��X�V

        //�V�[���̏�Ԃɉ������������s��
        switch (scene.buildIndex)
        {
            case 0: // �^�C�g���V�[��
                Debug.Log("�^�C�g���V�[�������[�h����܂���");
                break;
            case 1: // ���C���V�[��
                Debug.Log("���C���V�[�������[�h����܂���");
                break;
            case 2: // �X�g�[���[�V�[��
                Debug.Log("�X�g�[���[�V�[�������[�h����܂���");
                break;
        }
    }
}

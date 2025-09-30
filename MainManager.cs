using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System;
using System.Threading;
using Cysharp.Threading.Tasks;
using DG.Tweening;
using TMPro;
//��������'***'�ŕ\�L

public class MainManager : MonoBehaviour
{
    #region �V���A���C�Y�t�B�[���h
    [SerializeField] Vector2Int size; //�Q�[���̕��ƍ���
    [SerializeField] int bonusRate; //�{�[�i�X�̊|����
    [SerializeField] Vector2Int startPosition; //���q�J�n�ʒu
    [SerializeField] float dropTime; //��������
    [SerializeField] float continuousMoveTime; //�A���ړ�����
    [SerializeField] float downAcceleration; //��������
    [SerializeField] AtomObject atomPrefab; //���q�v���n�u
    [SerializeField] SpriteRenderer dropPointPrefab; //�����ʒu�v���n�u
    [SerializeField] int[] chainPointRates; //�A�����̓��_�{��
    [SerializeField] int[] comboPointRates; //�R���{���̓��_�{��
    [SerializeField] AtomType disturbanceAtom; //������܌��q
    [SerializeField] AtomType fullClearAtom; //�S�������q
    [SerializeField] float disturbancAtomsSize; //����܌��q�̃T�C�Y
    [SerializeField] int bonusAtomMin; //�{�[�i�X�̉��w���̌��q���̉���
    [SerializeField] float updateDropTime; //�������ԍX�V
    [SerializeField] float updateDropTimeRate; //�������ԍX�V��
    [SerializeField] GameObject stockAtoms; //���q�X�g�b�N
    [SerializeField] float formulaRate; //���������w���̓��_�{��
    [SerializeField] TextMeshProUGUI countdown; //�J�E���g�_�E��
    [SerializeField] RectTransform countdownRectTransform; //�J�E���g�_�E��
    #endregion �V���A���C�Y�t�B�[���h

    #region ���J�t�B�[���h
    public static MainManager Instance { get; private set; } //�C���X�^���X��ێ�����v���p�e�B
    public FormulaObject[] Formulas { get; private set; } = new FormulaObject[]
    {
        new FormulaObject("�t�b�����f", "HF", new Dictionary<AtomType, int> { { AtomType.H, 1 }, { AtomType.F, 1 } }),
        new FormulaObject("�������f", "HCl", new Dictionary<AtomType, int> { { AtomType.H, 1 }, { AtomType.Cl, 1 } }),
        new FormulaObject("�L�����f", "HBr", new Dictionary<AtomType, int> { { AtomType.H, 1 }, { AtomType.Br, 1 } }),
        new FormulaObject("���E�����f", "HI", new Dictionary<AtomType, int> { { AtomType.H, 1 }, { AtomType.I, 1 } }),
        new FormulaObject("�����i�g���E��", "NaCl", new Dictionary<AtomType, int> { { AtomType.Na, 1 }, { AtomType.Cl, 1 } }),
        new FormulaObject("�����J���E��", "KCl", new Dictionary<AtomType, int> { { AtomType.K, 1 }, { AtomType.Cl, 1 } }),
        new FormulaObject("������", "AgCl", new Dictionary<AtomType, int> { { AtomType.Ag, 1 }, { AtomType.Cl, 1 } }),
        new FormulaObject("������", "CuS", new Dictionary<AtomType, int> { { AtomType.Cu, 1 }, { AtomType.S, 1 } }),
        new FormulaObject("�����S", "FeS", new Dictionary<AtomType, int> { { AtomType.Fe, 1 }, { AtomType.S, 1 } }),
        new FormulaObject("��������", "ZnS", new Dictionary<AtomType, int> { { AtomType.Zn, 1 }, { AtomType.S, 1 } }),
        new FormulaObject("�����J���V�E��", "CaS", new Dictionary<AtomType, int> { { AtomType.Ca, 1 }, { AtomType.S, 1 } }),
        new FormulaObject("�_���}�O�l�V�E��", "MgO", new Dictionary<AtomType, int> { { AtomType.Mg, 1 }, { AtomType.O, 1 } }),
        new FormulaObject("�_���J���V�E��", "CaO", new Dictionary<AtomType, int> { { AtomType.Ca, 1 }, { AtomType.O, 1 } }),
        new FormulaObject("�_������", "ZnO", new Dictionary<AtomType, int> { { AtomType.Zn, 1 }, { AtomType.O, 1 } }),
        new FormulaObject("��_�����f", "NO", new Dictionary<AtomType, int> { { AtomType.N, 1 }, { AtomType.O, 1 } }),
        new FormulaObject("��_���Y�f", "CO", new Dictionary<AtomType, int> { { AtomType.C, 1 }, { AtomType.O, 1 } }),
        new FormulaObject("��", "H<sub>2</sub>O", new Dictionary<AtomType, int> { { AtomType.H, 2 }, { AtomType.O, 1 } }),
        new FormulaObject("�������f", "H<sub>2</sub>S", new Dictionary<AtomType, int> { { AtomType.H, 2 }, { AtomType.S, 1 } }),
        new FormulaObject("��_���Y�f", "CO<sub>2</sub>", new Dictionary<AtomType, int> { { AtomType.C, 1 }, { AtomType.O, 2 } }),
        new FormulaObject("�����}�O�l�V�E��", "MgCl<sub>2</sub>", new Dictionary<AtomType, int> { { AtomType.Mg, 1 }, { AtomType.Cl, 2 } }),
        new FormulaObject("�����J���V�E��", "CaCl<sub>2</sub>", new Dictionary<AtomType, int> { { AtomType.Ca, 1 }, { AtomType.Cl, 2 } }),
        new FormulaObject("��������", "ZnCl<sub>2</sub>", new Dictionary<AtomType, int> { { AtomType.Zn, 1 }, { AtomType.Cl, 2 } }),
        new FormulaObject("������(II)", "CuCl<sub>2</sub>", new Dictionary<AtomType, int> { { AtomType.Cu, 1 }, { AtomType.Cl, 2 } }),
        new FormulaObject("�����i�g���E��", "Na<sub>2</sub>S", new Dictionary<AtomType, int> { { AtomType.Na, 2 }, { AtomType.S, 1 } }),
        new FormulaObject("���_���i�g���E��", "NaOH", new Dictionary<AtomType, int> { { AtomType.Na, 1 }, { AtomType.O, 1 }, { AtomType.H, 1 } }),
        new FormulaObject("���_���J���E��", "KOH", new Dictionary<AtomType, int> { { AtomType.K, 1 }, { AtomType.O, 1 }, { AtomType.H, 1 } }),
        new FormulaObject("�����o���E��", "BaCl<sub>2</sub>", new Dictionary<AtomType, int> { { AtomType.Ba, 1 }, { AtomType.Cl, 2 } }),
        new FormulaObject("��_�����f", "NO<sub>2</sub>", new Dictionary<AtomType, int> { { AtomType.N, 1 }, { AtomType.O, 2 } }),
        new FormulaObject("��_������", "SO<sub>2</sub>", new Dictionary<AtomType, int> { { AtomType.S, 1 }, { AtomType.O, 2 } }),
        new FormulaObject("�ߎ_�����f", "H<sub>2</sub>O<sub>2</sub>", new Dictionary<AtomType, int> { { AtomType.H, 2 }, { AtomType.O, 2 } }),
        new FormulaObject("�A�����j�A", "NH<sub>3</sub>", new Dictionary<AtomType, int> { { AtomType.N, 1 }, { AtomType.H, 3 } }),
        new FormulaObject("�����A���~�j�E��", "AlCl<sub>3</sub>", new Dictionary<AtomType, int> { { AtomType.Al, 1 }, { AtomType.Cl, 3 } }),
        new FormulaObject("�Ɏ_", "HNO<sub>3</sub>", new Dictionary<AtomType, int> { { AtomType.H, 1 }, { AtomType.N, 1 }, { AtomType.O, 3 } }),
        new FormulaObject("�Ɏ_��", "AgNO<sub>3</sub>", new Dictionary<AtomType, int> { { AtomType.Ag, 1 }, { AtomType.N, 1 }, { AtomType.O, 3 } }),
        new FormulaObject("�Ɏ_�i�g���E��", "NaNO<sub>3</sub>", new Dictionary<AtomType, int> { { AtomType.Na, 1 }, { AtomType.N, 1 }, { AtomType.O, 3 } }),
        new FormulaObject("�Ɏ_�J���E��", "KNO<sub>3</sub>", new Dictionary<AtomType, int> { { AtomType.K, 1 }, { AtomType.N, 1 }, { AtomType.O, 3 } }),
        new FormulaObject("�Y�_�}�O�l�V�E��", "MgCO<sub>3</sub>", new Dictionary<AtomType, int> { { AtomType.Mg, 1 }, { AtomType.C, 1 }, { AtomType.O, 3 } }),
        new FormulaObject("�Y�_�J���V�E��", "CaCO<sub>3</sub>", new Dictionary<AtomType, int> { { AtomType.Ca, 1 }, { AtomType.C, 1 }, { AtomType.O, 3 } }),
        new FormulaObject("�Y�_�o���E��", "BaCO<sub>3</sub>", new Dictionary<AtomType, int> { { AtomType.Ba, 1 }, { AtomType.C, 1 }, { AtomType.O, 3 } }),
        new FormulaObject("���_���J���V�E��", "Ca(OH)<sub>2</sub>", new Dictionary<AtomType, int> { { AtomType.Ca, 1 }, { AtomType.O, 2 }, { AtomType.H, 2 } }),
        new FormulaObject("���_���o���E��", "Ba(OH)<sub>2</sub>", new Dictionary<AtomType, int> { { AtomType.Ba, 1 }, { AtomType.O, 2 }, { AtomType.H, 2 } }),
        new FormulaObject("�_���A���~�j�E��", "Al<sub>2</sub>O<sub>3</sub>", new Dictionary<AtomType, int> { { AtomType.Al, 2 }, { AtomType.O, 3 } }),
        new FormulaObject("�_���S(III)", "Fe<sub>2</sub>O<sub>3</sub>", new Dictionary<AtomType, int> { { AtomType.Fe, 2 }, { AtomType.O, 3 } }),
        new FormulaObject("���^��", "CH<sub>4</sub>", new Dictionary<AtomType, int> { { AtomType.C, 1 }, { AtomType.H, 4 } }),
        new FormulaObject("�Y�_", "H<sub>2</sub>CO<sub>3</sub>", new Dictionary<AtomType, int> { { AtomType.H, 2 }, { AtomType.C, 1 }, { AtomType.O, 3 } }),
        new FormulaObject("�����A�����j�E��", "NH<sub>4</sub>Cl", new Dictionary<AtomType, int> { { AtomType.N, 1 }, { AtomType.H, 4 }, { AtomType.Cl, 1 } }),
        new FormulaObject("���_��(II)", "CuSO<sub>4</sub>", new Dictionary<AtomType, int> { { AtomType.Cu, 1 }, { AtomType.S, 1 }, { AtomType.O, 4 } }),
        new FormulaObject("���_�S(II)", "FeSO<sub>4</sub>", new Dictionary<AtomType, int> { { AtomType.Fe, 1 }, { AtomType.S, 1 }, { AtomType.O, 4 } }),
        new FormulaObject("���_�J���V�E��", "CaSO<sub>4</sub>", new Dictionary<AtomType, int> { { AtomType.Ca, 1 }, { AtomType.S, 1 }, { AtomType.O, 4 } }),
        new FormulaObject("�Y�_�i�g���E��", "Na<sub>2</sub>CO<sub>3</sub>", new Dictionary<AtomType, int> { { AtomType.Na, 2 }, { AtomType.C, 1 }, { AtomType.O, 3 } }),
        new FormulaObject("�G�`����", "C<sub>2</sub>H<sub>4</sub>", new Dictionary<AtomType, int> { { AtomType.C, 2 }, { AtomType.H, 4 } }),
        new FormulaObject("���_", "H<sub>2</sub>SO<sub>4</sub>", new Dictionary<AtomType, int> { { AtomType.H, 2 }, { AtomType.S, 1 }, { AtomType.O, 4 } }),
        new FormulaObject("���_�i�g���E��", "Na<sub>2</sub>SO<sub>4</sub>", new Dictionary<AtomType, int> { { AtomType.Na, 2 }, { AtomType.S, 1 }, { AtomType.O, 4 } }),
        new FormulaObject("���_�J���E��", "K<sub>2</sub>SO<sub>4</sub>", new Dictionary<AtomType, int> { { AtomType.K, 2 }, { AtomType.S, 1 }, { AtomType.O, 4 } }),
        new FormulaObject("���_���A���~�j�E��", "Al(OH)<sub>3</sub>", new Dictionary<AtomType, int> { { AtomType.Al, 1 }, { AtomType.O, 3 }, { AtomType.H, 3 } }),
        new FormulaObject("�|�_", "CH<sub>3</sub>COOH", new Dictionary<AtomType, int> { { AtomType.C, 2 }, { AtomType.H, 4 }, { AtomType.O, 2 } }),
        new FormulaObject("�����_", "H<sub>3</sub>PO<sub>4</sub>", new Dictionary<AtomType, int> { { AtomType.H, 3 }, { AtomType.P, 1 }, { AtomType.O, 4 } }),
        new FormulaObject("�|�_�i�g���E��", "CH<sub>3</sub>COONa", new Dictionary<AtomType, int> { { AtomType.C, 2 }, { AtomType.H, 3 }, { AtomType.O, 2 }, { AtomType.Na, 1 } }),
        new FormulaObject("�G�^��", "C<sub>2</sub>H<sub>6</sub>", new Dictionary<AtomType, int> { { AtomType.C, 2 }, { AtomType.H, 6 } }),
        new FormulaObject("�Ɏ_��(II)", "Cu(NO<sub>3</sub>)<sub>2</sub>", new Dictionary<AtomType, int> { { AtomType.Cu, 1 }, { AtomType.N, 2 }, { AtomType.O, 6 } }),
        new FormulaObject("�Ɏ_�J���V�E��", "Ca(NO<sub>3</sub>)<sub>2</sub>", new Dictionary<AtomType, int> { { AtomType.Ca, 1 }, { AtomType.N, 2 }, { AtomType.O, 6 } }),
        new FormulaObject("�v���s����", "C<sub>3</sub>H<sub>6</sub>", new Dictionary<AtomType, int> { { AtomType.C, 3 }, { AtomType.H, 6 } }),
        new FormulaObject("�v���p��", "C<sub>3</sub>H<sub>8</sub>", new Dictionary<AtomType, int> { { AtomType.C, 3 }, { AtomType.H, 8 } }),
        new FormulaObject("�Y�_�A�����j�E��", "(NH<sub>4</sub>)<sub>2</sub>CO<sub>3</sub>", new Dictionary<AtomType, int> { { AtomType.N, 2 }, { AtomType.H, 8 }, { AtomType.C, 1 }, { AtomType.O, 3 } }),
    }; //���w���̔z��
    [HideInInspector] public event Action<AtomType> onCreateAtomType; //���q�����C�x���g
    #endregion ���J�t�B�[���h

    #region �v���C�x�[�g�t�B�[���h 
    int[] atomWeight; //���q�̏d��
    State currentState; //���݂̏�Ԃ��Ǘ�����ϐ�
    int fullWeight; //�d�݂̍��v
    PlayerBase[] players; //�v���C���[
    float gameTime; //�Q�[������
    FormulaObject bonusFormula; //�{�[�i�X���w��
    float updateDropCount; //�����X�V�J�E���g
    CancellationTokenSource cts; //�L�����Z���g�[�N��
    readonly Color disturbanceAtomColor = new Color32(0x22, 0x22, 0x22, 255);    //������ܐF
    static readonly Dictionary<AtomGroupType, Color> groupColors = new Dictionary<AtomGroupType, Color>
    {
        { AtomGroupType.Hydrogen,           new Color32(0x00, 0xCF, 0xFF, 255) },
        { AtomGroupType.Halogen,            new Color32(0x00, 0xAA, 0x88, 255) },
        { AtomGroupType.OxygenSulfur,       new Color32(0xA6, 0xE2, 0x2E, 255) },
        { AtomGroupType.NitrogenPhosphorus, new Color32(0xB3, 0x9D, 0xDB, 255) },
        { AtomGroupType.Carbon,             new Color32(0x44, 0x44, 0x44, 255) },
        { AtomGroupType.AlkaliMetal,        new Color32(0xFF, 0xD7, 0x00, 255) },
        { AtomGroupType.AlkalineEarthMetal, new Color32(0xD1, 0xC4, 0xE9, 255) },
        { AtomGroupType.TransitionMetal,    new Color32(0x3F, 0x51, 0xB5, 255) },
        { AtomGroupType.SimpleMetal,        new Color32(0xB0, 0xBE, 0xC5, 255) },
        { AtomGroupType.None,               Color.clear }
    }; //���q�̐F���Ǘ����鎫��
    #endregion �v���C�x�[�g�t�B�[���h

    #region �萔
    public const int NormalMax = 8; //����ܕ��ʕ\���̍ő�l
    #endregion �萔

    #region �v���p�e�B
    public State CurrentState //���݂̏�Ԃ��擾�܂��͐ݒ肷��v���p�e�B
    {
        get 
        {
            return currentState; 
        }
        private set 
        { 
            if(currentState == value) return; //��Ԃ��ς��Ȃ��ꍇ�͉������Ȃ�
            currentState = value; 
            switch(currentState)
            {
                case State.Ready:
                    // Ready��Ԃ̏���
                    Countdown().Forget();
                    break;
                case State.Play:
                    // Play��Ԃ̏���
                    break;
                case State.Pause:
                    // Pause��Ԃ̏���
                    break;
                case State.Result:
                    // Result��Ԃ̏���
                    break;
                default:
                    break;
            }
        }
    }
    public AtomType DisturbanceAtom => disturbanceAtom; //����܌��q���
    public AtomType FullClearAtom => fullClearAtom; //�S�������q���
    public int Level { get; set; } //�Q�[���̃��x�����Ǘ�����v���p�e�B
    public int ChainPointRatesLength //�A�����̒���
    {
        get
        {
            return chainPointRates.Length;
        }
    }
    public int ComboPointRatesLength //�R���{���̒���
    {
        get
        {
            return comboPointRates.Length;
        }
    }
    public float DisturbancAtomsSize => disturbancAtomsSize; //����܌��q�̃T�C�Y
    public List<AtomObject> DisturbanceAtomObjects { get; set; } //������܌��q���Ǘ�����z��
    public int StageSize => size.x * size.y; //�X�e�[�W�T�C�Y
    public Color DisturbanceAtomColor => disturbanceAtomColor; //�X�e�[�W�̕�
    public SpriteRenderer DropPointPrefab => dropPointPrefab; //�����ʒu�v���n�u
    public Vector2Int Size => size; //�Q�[���̕��ƍ���
    public Vector2Int StartPosition => startPosition; //���q�J�n�ʒu
    public float DropTime => dropTime; //��������
    public float ContinuousMoveTime => continuousMoveTime; //�A���ړ�����
    public float DownAcceleration => downAcceleration; //��������
    public AtomObject AtomPrefab => atomPrefab; //���q�v���n�u
    public GameObject StockAtoms => stockAtoms; //���q�X�g�b�N
    #endregion �v���p�e�B

    #region �񋓑�
    public enum State
    {
        None,
        Ready,
        Play,
        Pause,
        Result,
    } //�Q�[���̏�Ԃ��Ǘ�����񋓌^
    public enum AtomType
    {
        H, 
        Cl,
        Br,
        F,
        I,
        O,
        S,
        N,
        P,
        C,  
        K,
        Na,
        Ba,
        Ca,
        Mg,
        Cu,
        Zn,
        Ag,
        Fe,
        Al,
        None,
    } //���q�̎�ނ��Ǘ�����񋓌^
    public enum AtomGroupType //���q�̃O���[�v���Ǘ�����񋓌^
    {
        Hydrogen = 1,
        Halogen,
        OxygenSulfur,
        NitrogenPhosphorus,
        Carbon,
        AlkaliMetal,
        AlkalineEarthMetal,
        TransitionMetal,
        SimpleMetal,
        None
    }
    #endregion �񋓑�

    #region Unity�C�x���g
    private void Awake()
    {
        Instance = this;
    }
    private void Start()
    {
        //�Q�[���̏�����
        Init();
        for(int i = 0; i < players.Length; i++)
        {
            players[i].Start0(i); // �v���C���[�̏�����
        }
        CurrentState = State.Ready; //�Q�[���̏�Ԃ�������
    }
    private void Update()
    {
        switch (CurrentState)
        {
            case State.Ready:
                // Ready��Ԃ̏���
                //���̌��q�܂ł��Z�b�g
                foreach (var player in players)
                {
                    player.ReadyAtom(); //���q���Z�b�g
                }   
                break;
            case State.Play:
                // Play��Ԃ̏���
                UpdateGameTime(); //�Q�[�����Ԃ��X�V
                foreach (var player in players)
                {
                    player.UpdatePlayState(); // �v���C���[�̏�Ԃ��X�V
                }
                break;
            case State.Pause:
                // Pause��Ԃ̏���
                break;
            case State.Result:
                // Result��Ԃ̏���
                break;
            default:
                break;
        }
    }
    private void OnDestroy()
    {
        //�L�����Z���g�[�N�����L�����Z��
        if (cts != null)
        {
            cts.Cancel();
            cts.Dispose();
            cts = null;
        }
    } //�^�X�N�̃L�����Z��
    #endregion Unity�C�x���g

    #region ���J���\�b�h 
    public void CreateAtoms() //�y�A���q�𐶐����郁�\�b�h
    {
        //�d�݂ɉ����Č��q�̎�ނ�����
        int atomCount = UnityEngine.Random.Range(0, fullWeight); //���q�̎�ނ������_���ɑI��
        for (int j = 0; j < (int)AtomType.None; j++)
        {
            if (atomWeight[j] > atomCount)
            {
                onCreateAtomType.Invoke((AtomType)j); //���q�����C�x���g�𔭉�
                break;
            }
            atomCount -= atomWeight[j];
        }
    }
    public void CalcPoint(ref PointSet pointSet, int atomCount,�@int formulaPoint, int chainCount, int comboCount) //���_�̌v�Z
    {
        //���_�̌v�Z
        chainCount = Mathf.Min(chainCount, ChainPointRatesLength - 1); //�A�������X�V
        comboCount = Mathf.Min(comboCount, ComboPointRatesLength - 1); //�R���{�����X�V
        pointSet.AtomCount = atomCount; //���q�̐����Z�b�g
        pointSet.FormulaPoint = formulaPoint; //���w���̓��_���Z�b�g
        pointSet.ChainRate = chainPointRates[chainCount]; //�A�����ɉ������{��
        pointSet.ComboRate = comboPointRates[comboCount]; //�R���{���ɉ������{��
        pointSet.Point = atomCount * formulaPoint * pointSet.ChainRate * pointSet.ComboRate; //���_���v�Z
    } 
    public void PassDisturbance(int num, PlayerBase player, Vector3 center)
    {
        if (players[0] == player) players[1].GotDisturbanceNumber(num, center).Forget(); //�v���C���[1�̂���܌��q����ݒ�
        else players[0].GotDisturbanceNumber(num, center).Forget(); //�v���C���[0�̂���܌��q����ݒ� 
    }
    #endregion ���J���\�b�h

    #region ����J���\�b�h
    private void Init() //�Q�[���̏�����
    {
        cts = new CancellationTokenSource(); //�L�����Z���g�[�N����������
        gameTime = 0f; //�Q�[�����Ԃ�������
        players = FindObjectsByType<PlayerBase>(FindObjectsSortMode.None); //�v���C���[���擾
        int[] atomCount = new int[(int)AtomType.None]; //���q�̐���������
        foreach (var formula in Formulas)
        {
            int count = 0; //���q�̐���������
            foreach (var atom in formula.AtomDictionary)
            {
                count += atom.Value; //�����q�̐����J�E���g
                atomCount[(int)atom.Key] += atom.Value; //���q�̐����J�E���g
            }
        }
        //�{�[�i�X���w�������߂�
        //atomCount�̐���bonusFormula�̐����傫�����w������AbonusFormula�������_���ɑI��
        var selectedList = Formulas.Where(a => a.AtomCount >= bonusAtomMin).ToArray();
        if (selectedList.Length > 0)
        {
            var bonusIndex = UnityEngine.Random.Range(0, selectedList.Length); // Unity��Random
            bonusFormula = selectedList[bonusIndex]; //�{�[�i�X���w����I��
        }
        WeightPerLevel(atomCount); //���q�̏d�݂����x���ɉ����Đݒ�
        DisturbanceAtomObjects = new List<AtomObject>(20);
        for (int i = 0; i < 20; i++)
        {
            InstantiateDisturbance();
        }
    }
    public void InstantiateDisturbance()
    {
        AtomObject atomObject = Instantiate(atomPrefab, stockAtoms.transform); //���q���C���X�^���X��
        atomObject.Set(disturbanceAtomColor, disturbanceAtom);
        atomObject.UnEnabled();
        atomObject.transform.localScale *= DisturbancAtomsSize; //������܌��q�̃T�C�Y��ύX
        DisturbanceAtomObjects.Add(atomObject);
    }
    private void WeightPerLevel(in int[] atomCount)
    {
        atomWeight = new int[(int)AtomType.None];
        //�����d�݂�������
        fullWeight = 0;
        for (int i = 0; i < (int)AtomType.None; i++)
        {
            if (i == (int)disturbanceAtom) continue; //������܌��q�̏d�݂�0�ɂ���
            //���q�̏d�݂����x���ɉ����Đݒ�
            if (Level == 0)
            {
                atomWeight[i] = atomCount[i] switch
                {
                    > 60 => 32,
                    > 13 => 16,
                    > 8 => 8,
                    > 3 => 4,
                    > 2 => 2,
                    _ => 1,
                };
            }
            else if(Level == 1)
            {
                atomWeight[i] = atomCount[i] switch
                {
                    > 60 => 16,
                    > 13 => 8,
                    > 8 => 4,
                    > 3 => 2,
                    _ => 1,
                };
            }
            else
            {
                atomWeight[i] = atomCount[i] switch
                {
                    > 60 => 16,
                    > 8 => 8,
                    > 4 => 4,
                    > 2 => 2,
                    _ => 1,
                };
            }
            //�����d�݂��v�Z
            fullWeight += atomWeight[i];
        }
    } //���q�̏d�݂����x���ɉ����Đݒ肷�郁�\�b�h
    static public Color GetAtomColor(AtomType atomType)
    {
        var atomGroup = AtomGroupHelper.atomGroups[atomType]; //���q�̃O���[�v���擾
        return groupColors[atomGroup];
    } //���q�̐F���擾
    public void Pause()
    {
        if (CurrentState == State.Pause)
        {
            CurrentState = State.Play;
        }
        else if(CurrentState == State.Play)
        {
            CurrentState = State.Pause;
        }
    } //�|�[�Y
    async UniTaskVoid Countdown()
    {
        //�J�E���g�_�E������
        for (int i = 0; i < players.Length; i++)
        {
            players[i].PlayCountdownAsync().Forget(); //�J�E���g�_�E���̃e�L�X�g���Z�b�g
        }
        await UniTask.WaitForSeconds(3f, ignoreTimeScale: false, cancellationToken: cts.Token); //�J�E���g�_�E���̑ҋ@
        CurrentState = State.Play; //�J�E���g�_�E���I����A��Ԃ�Play�ɕύX
        for (int i = 0; i < players.Length; i++)
        {
            players[i].SetStart();
        }
    } //�J�E���g�_�E�����s���R���[�`��   
    void UpdateGameTime()
    {
        // Play��Ԃ̏���
        gameTime += Time.deltaTime; //�Q�[�����Ԃ��X�V
        updateDropCount += Time.deltaTime;
        if(updateDropCount >= updateDropTime)
        {
            dropTime *= updateDropTimeRate;
            updateDropCount -= updateDropTime;
        }
    } //�Q�[�����Ԃ̍X�V
    #endregion ����J���\�b�h

    #region ���J�N���X
    public class FormulaObject //���w�����Ǘ�����N���X
    {
        public string Name { get; private set; } //���w���̖��O
        public string Formula { get; private set; }//���w��
        public Dictionary<AtomType, int> AtomDictionary { get; private set; } //���q�̎�ނƂ��̐����Ǘ����鎫��
        public int AtomCount { get; private set; }//���q�̐�
        public int Point { get; private set; } //���_
        public FormulaObject(string name, string formula, Dictionary<AtomType, int> atomDict)
        {
            Name = name;�@//���w���̖��O��ݒ�
            Formula = formula;�@//���w����ݒ�
            AtomDictionary = atomDict;�@//���q�̎�ނƂ��̐���ݒ�

            AtomCount = atomDict.Values.Sum(); //���q�̐����J�E���g
            Point = atomDict.Sum(pair =>
            {
                var group = AtomGroupHelper.GetGroup(pair.Key);
                var multiplier = group switch
                {
                    AtomGroupType.AlkaliMetal or
                    AtomGroupType.AlkalineEarthMetal or
                    AtomGroupType.TransitionMetal or
                    AtomGroupType.SimpleMetal => 3,
                    _ => 1
                };
                return pair.Value * multiplier;
            });
        }
    }
    public static class AtomGroupHelper
    {
        public static readonly Dictionary<AtomType, AtomGroupType> atomGroups = new Dictionary<AtomType, AtomGroupType>
    {
        { AtomType.H, AtomGroupType.Hydrogen },
        { AtomType.Cl, AtomGroupType.Halogen },
        { AtomType.F,  AtomGroupType.Halogen },
        { AtomType.Br, AtomGroupType.Halogen },
        { AtomType.I,  AtomGroupType.Halogen },
        { AtomType.O,  AtomGroupType.OxygenSulfur },
        { AtomType.S,  AtomGroupType.OxygenSulfur },
        { AtomType.N,  AtomGroupType.NitrogenPhosphorus },
        { AtomType.P,  AtomGroupType.NitrogenPhosphorus },
        { AtomType.C,  AtomGroupType.Carbon },
        { AtomType.Na, AtomGroupType.AlkaliMetal },
        { AtomType.K,  AtomGroupType.AlkaliMetal },
        { AtomType.Mg, AtomGroupType.AlkalineEarthMetal },
        { AtomType.Ca, AtomGroupType.AlkalineEarthMetal },
        { AtomType.Ba, AtomGroupType.AlkalineEarthMetal },
        { AtomType.Fe, AtomGroupType.TransitionMetal },
        { AtomType.Cu, AtomGroupType.TransitionMetal },
        { AtomType.Zn, AtomGroupType.TransitionMetal },
        { AtomType.Ag, AtomGroupType.TransitionMetal },
        { AtomType.Al, AtomGroupType.SimpleMetal },
        { AtomType.None, AtomGroupType.None }
    };

        public static AtomGroupType GetGroup(AtomType atomType)
        {
            return atomGroups.TryGetValue(atomType, out var group) ? group : AtomGroupType.None;
        }
    } //���q���Ǘ�����N���X
    public struct PointSet //���_���Ǘ�����\����
    {
        public int Point; //���_
        public int FormulaPoint; //��b���_
        public int AtomCount; //���q��
        public int ChainRate; //�A����
        public int ComboRate; //�R���{��
    }
    #endregion ���J�N���X
}

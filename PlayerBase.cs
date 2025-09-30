using System.Collections.Generic;
using TMPro;
using UnityEngine;
using System;
using static MainManager;
using System.Threading;
using System.Linq;
using DG.Tweening;
using Cysharp.Threading.Tasks;
using System.Text;
using UnityEngine.Pool;
//��������'***'�ŕ\�L

public class PlayerBase : MonoBehaviour
{
    #region �V���A���C�Y�t�B�[���h
    [SerializeField] TextMeshProUGUI countdownText; //�J�E���g�_�E���p�̃e�L�X�g
    [SerializeField] TextMeshProUGUI formulaText; //���w���e�L�X�g
    [SerializeField] TextMeshProUGUI chemicalText; //���w���e�L�X�g
    [SerializeField] TextMeshProUGUI pointText; //���_�e�L�X�g
    [SerializeField] TextMeshProUGUI formulaPointText; //�e���_�e�L�X�g
    [SerializeField] Vector2 leftBottomPosition ;//�����̈ʒu
    [SerializeField] Vector2[] nextAtomsPosition; //���̌��q�̈ʒu
    [SerializeField] Vector3 disturbancAtomPosition; //����܌��q�̈ʒu
    #endregion �V���A���C�Y�t�B�[���h

    #region �v���C�x�[�g�t�B�[���h
    Vector3 formulaStart; //���w���\�������ʒu
    Vector3 chemicalStart; //���w���\�������ʒu
    protected AtomObject[] nextAtoms; //���̌��q���Ǘ�����z��
    protected PlayState playState; //�v���C�̏�
    protected AtomObject[] currentAtoms; //���݂̌��q���Ǘ�����z��
    protected Vector2Int[] currentAtomPositions; //���݂̌��q�̈ʒu
    int point; //�v���C���[�̓��_
    int chainCount; //�A����
    int comboCount; //�R���{��
    AtomObject[,] stageAtom; //���q�̈ʒu���Ǘ�����z��
    protected float[] stageAtomF; //���q�ԍ��̈ʒu���Ǘ�����z��
    Vector2Int[] dropVector2Ints; //�h���b�v�\���ʒu
    protected int[] dropAtomYs; //��ԏ�̌��q
    List<Vector2Int> checkBuffer; //�T�������q�o�b�t�@
    Dictionary<FormulaObject, HashSet<Vector2Int>> atomObjectHashs; //���������q�̃��X�g
    Queue<AtomType> atomsQueue; //�L���[�y�A���q
    float dropCount; //�����J�E���g
    int disturbancStartX; //����܃X�^�[�g�ʒu
    CancellationTokenSource cts; //�L�����Z���g�[�N��
    SpriteRenderer[] dropPoints; //�����n�_�I�u�W�F�N�g
    PointSet pointSet = new PointSet(); //���_���Ǘ�����
    ObjectPool<AtomObject> atomObjectPool; //���q�̃v�[��
    StringBuilder stringBuilder; //�e�L�X�g
    Vector2Int[] vector2Ints; //�ړ���
    int passTotalDisturbanceNumber; //�n��������܌��q��
    protected int gotDisturbanceNumber; //�擾��������܌��q��
    #endregion �v���C�x�[�g�t�B�[���h

    #region �v���e�N�g�t�B�[���h
    /// <summary> ���C���}�l�[�W���[ </summary>
    protected MainManager M { get; private set; } //���C���}�l�[�W���[
    #endregion �v���e�N�g�t�B�[���h

    #region �萔
    const int displayUpper = 12; //�X�e�[�W�̕\���ő卂��
    readonly Color countdownTextColor = Color.white; //�J�E���g�_�E���e�L�X�g�J���[
    #endregion �萔

    #region �v���p�e�B    
    #endregion �v���p�e�B

    #region �񋓑�
    protected enum PlayState //�v���C�J��
    {
        None,
        Move,
        NoMove,
        GameOver,
    }
    public enum Direction //����
    {
        Right,
        Left,
        Down,
        None,
    }
    public enum RotationInfo //��]���
    {
        Top,
        Right,
        Down,
        Left,
    }
    #endregion �񋓑�

    #region ���J���\�b�h
    public virtual void Start0(int id) //�v���C���[�̏��������s�����\�b�h
    {
        M = Instance; //MainManager�̃C���X�^���X���擾
        //�Q�[���̏�����
        Init();
    }
    public void ReadyAtom() //���q�̏���
    {
        //���̌��q�܂ł��Z�b�g
        if (nextAtoms[0] == null)
        {
            SetAtoms(false); //���q���Z�b�g
        }
    }
    public void UpdatePlayState() //Play��Ԃ̏������s�����\�b�h
    {
        if (playState == PlayState.Move) PlayMove();
    }
    public AtomObject SetAtom(AtomType atomType, PlayerBase player) //���݌��q���Z�b�g
    {
        //AtomObject atomObject = atomObjects.First(x => x.IsExist == false);
        AtomObject atomObject = atomObjectPool.Get();
        var atomGroup = AtomGroupHelper.atomGroups[atomType]; //���q�̃O���[�v���擾
        atomObject.Set(GetAtomColor(atomType), atomType);
        atomObject.transform.parent = player.transform; //�e���v���C���[�ɐݒ�
        return atomObject;
    }
    public void SetPoint(int atomCount, int formulaPoint, int chainCount, int comboCount) //���_�̌v�Z
    {
        M.CalcPoint(ref pointSet, atomCount, formulaPoint, chainCount, comboCount);
        point += pointSet.Point;
    }
    public async UniTaskVoid GotDisturbanceNumber(int num, Vector3 center) //����܌��q���Z�b�g���郁�\�b�h
    {
        Vector3 vector3 = disturbancAtomPosition;
        Tween tween = null;
        var startIndex = gotDisturbanceNumber;
        gotDisturbanceNumber += num;
        var endIndex = gotDisturbanceNumber;
        if (startIndex > 0)
        {
            vector3.x = M.DisturbanceAtomObjects[startIndex - 1].transform.localPosition.x + M.DisturbancAtomsSize;
        }
        for (int i = startIndex; i < endIndex; i++)
        {
            if (M.DisturbanceAtomObjects.Count <= i) M.InstantiateDisturbance();
            M.DisturbanceAtomObjects[i].Enabled(); //����܌��q��ǉ�
            M.DisturbanceAtomObjects[i].transform.parent = transform; //�e���v���C���[�ɐݒ�
            M.DisturbanceAtomObjects[i].transform.position = center;
            tween = M.DisturbanceAtomObjects[i].transform.DOLocalMove(vector3, 0.5f);
            vector3.x += M.DisturbancAtomsSize;
        }
        await tween?.AsyncWaitForCompletion();
        if (NormalMax < gotDisturbanceNumber)
        {
            vector3.x = M.DisturbanceAtomObjects[0].transform.localPosition.x;
            for (int i = 1; i < gotDisturbanceNumber; i++)
            {
                vector3.x += (M.DisturbancAtomsSize * (NormalMax - 1)) / (gotDisturbanceNumber - 1);
                M.DisturbanceAtomObjects[i].transform.DOLocalMove(vector3, 0.5f);
            }
        }
    }
    #endregion ���J���\�b�h

    #region ����J���\�b�h     
    protected void Init() //�v���C���[�̏�����
    {
        vector2Ints = new Vector2Int[2]; //�ړ��ʂ�������
        dropVector2Ints = new Vector2Int[2]; //�h���b�v�n�_��������
        cts = new CancellationTokenSource(); //�L�����Z���g�[�N����������
        M.onCreateAtomType += type => atomsQueue.Enqueue(type); //���q�̃L���[���쐬
        atomObjectHashs = new Dictionary<FormulaObject, HashSet<Vector2Int>>(); //���������q�̃��X�g��������
        atomsQueue = new Queue<AtomType>(); //�y�A���q�̃L���[��������
        stageAtom = new AtomObject[M.Size.x, M.Size.y]; //���q�̈ʒu��������
        stageAtomF = new float[M.Size.x * M.Size.y]; //���q�ԍ��̈ʒu��������
        dropAtomYs = new int[M.Size.x];
        nextAtoms = new AtomObject[4]; //���̌��q��������
        currentAtoms = new AtomObject[2]; //���݂̌��q��������
        currentAtomPositions = new Vector2Int[2];
        checkBuffer = new List<Vector2Int>(23); //�T�������q�o�b�t�@��������
        //�h���b�v�n�_��������
        dropPoints = new SpriteRenderer[2];
        for(int i = 0; i< dropPoints.Length; i++)        
        {
            dropPoints[i] = Instantiate(M.DropPointPrefab, transform); //�h���b�v�n�_�𐶐�
            dropPoints[i].enabled = false; //�h���b�v�n�_���\���ɂ���
        }
        atomObjectPool = new ObjectPool<AtomObject>(
           createFunc: () => Instantiate(M.AtomPrefab),
           actionOnGet:go =>
           {
               go.Enabled();
               go.transform.parent = transform; //�e���v���C���[�ɐݒ�
           },
           actionOnRelease: go => 
           {
               go.UnEnabled();
               go.transform.parent = M.StockAtoms.transform; //�e��null�ɐݒ�
           },
           collectionCheck: false,
           defaultCapacity: 100,
           maxSize: M.StageSize + 10
        );
        formulaPointText.enabled = false;
        formulaText.enabled = false;
        chemicalText.enabled = false;
        formulaStart = formulaText.rectTransform.position;
        chemicalStart = chemicalText.rectTransform.position;
        pointText.text = $"{point.ToString()}P" ;
        stringBuilder = new StringBuilder();
    }
    protected virtual void SetAtoms(bool isStart = true) //���q���Z�b�g���郁�\�b�h
    {
        //���q���ړ�
        for (int i = 0; i < 2; i++)
        {
            //���݂̌��q�̐ݒ�
            currentAtoms[i] = nextAtoms[i]; //���̌��q�����݂̌��q�ɐݒ�
            if (currentAtoms[i] != null)
            {
                var vector = new Vector2Int(M.StartPosition.x, M.StartPosition.y + i);
                Set(currentAtoms[i], vector, i); //���q���ړ�
            }
            //���̌��q�̐ݒ�
            nextAtoms[i] = nextAtoms[i + 2]; //���̎��̌��q�����̌��q�ɐݒ�
            if (nextAtoms[i] != null)
            {
                nextAtoms[i].transform.localPosition = new Vector3(nextAtomsPosition[0].x, nextAtomsPosition[0].y + i, 0f); //���q�̈ʒu��ݒ�
            }
            //���̎��̌��q�̐ݒ�
            if (atomsQueue.Count <= 0)
            {
                M.CreateAtoms(); //�y�A���q�𐶐�
            }
            nextAtoms[i + 2] = SetAtom(atomsQueue.Dequeue(), this); //���q���Z�b�g
            nextAtoms[i + 2].transform.localPosition = new Vector3(nextAtomsPosition[1].x, nextAtomsPosition[1].y + i, 0f); ;
        }
        if (isStart)
        {
            playState = PlayState.Move;
            SetDropPoint(ref dropVector2Ints, ref currentAtomPositions);
            //�h���b�v�J�E���g�𔼕��ɂ���
            dropCount = M.DropTime / 2;
        }
    }  
    private void Set(AtomObject atomObject, in Vector2Int currect, int index) //���q���Z�b�g
    {
        if (stageAtom[currect.x, currect.y] != null)
        {
            playState = PlayState.GameOver;
        }
        //���q���ړ�
        atomObject.transform.localPosition = currect + leftBottomPosition; //Transform�̈ʒu��ݒ�
        SettingDisplay(currect.y, atomObject);
        currentAtomPositions[index] = currect;
    }    
    void SetFreeAtom(int x, int y, AtomObject atomObject) //���q��ݒ肷�郁�\�b�h
    {
        var vector2Int = new Vector2Int(x, y);
        stageAtom[x, y] = atomObject; //���q�̈ʒu��ݒ�
        stageAtomF[x + y * M.Size.x] = (int)atomObject.AtomType + 1; //���q�ԍ���ݒ�
        stageAtom[x, y].transform.localPosition = vector2Int + leftBottomPosition; //Transform�̈ʒu��ݒ�
    }
    protected virtual void PlayMove() //�ړ����͂��������郁�\�b�h
    {
#if DEBUG
        DropStep(Time.deltaTime * 0.3f); //�h���b�v����
#else
        DropStep(Time.deltaTime); //�h���b�v����
#endif
    }
    protected void DropStep(float deltaTime) //�ړ����͂��������郁�\�b�h
    {
        //�h���b�v�J�E���g���X�V
        dropCount += deltaTime;
        //�h���b�v�J�E���g����莞�Ԃ𒴂����牺�ړ�
        if (dropCount >= M.DropTime)
        {
            bool move = Move(Direction.Down);
            dropCount = 0; //�h���b�v�J�E���g�����Z�b�g
            if (!move)
            {
                foreach(var dropPoint in dropPoints)
                {
                    dropPoint.enabled = false; //�h���b�v�n�_���\���ɂ���
                }
                EndCheck().Forget();
            }
        }
    }
    async UniTaskVoid EndCheck() //�`�F�b�N�����ׂĂ��炩����
    {
        playState = PlayState.NoMove;
        await FreeFall();
        await CheckOrFullClear();
        var isEnd = !DropGotDisturbanceNumber();
        if (!isEnd)
        {
            await FreeFall(true);
            await CheckOrFullClear();
        }
        SetAtoms();
    }
    async UniTask CheckOrFullClear() //�`�F�b�N�����ׂĂ��炩����
    {
        chainCount = 0;
        comboCount = 0;
        while (true)
        {
            if (Check())
            {
                //���w�����`�F�b�N����
                var center = await PointCount(); //���_���v�Z
                await AtomDelete(); //���q���폜
                await FreeFall(true);
                SetDisturbance(center);
            }
            else
            {
                break;
            }
            bool allNull = true;
            //��ԉ���null���ǂ���
            for (int i = 0; i < M.Size.x; i++)
            {
                if (stageAtom[i, 0] != null)
                {
                    allNull = false;
                    break;
                }
            }
            if (allNull)
            {
                //���ׂĂ̌��q���Ȃ�
                await FullClear();
                break;
            }
        }
    }
    public async UniTaskVoid PlayCountdownAsync() //�J�E���g�_�E�����\�b�h
    {
        string[] texts = { "3", "2", "1", "Go!" };
        foreach (var text in texts)
        {
            var end = text == "Go!";
            countdownText.text = text;
            countdownText.color = countdownTextColor;
            countdownText.transform.localScale = Vector3.zero;
            var sequence = DOTween.Sequence();
            sequence.Append(countdownText.transform.DOScale(1.5f, 0.2f).SetEase(end ? Ease.OutSine : Ease.OutBack));
            sequence.Join(countdownText.DOFade(1f, 0.1f));
            if(!end) sequence.AppendInterval(0.5f);
            sequence.Append(countdownText.DOFade(0f, end ? 1.4f : 0.3f).SetEase(end ? Ease.InOutSine : Ease.Unset));
            sequence.Join(countdownText.transform.DOScale(end ? 2.4f : 2f, end ? 1.4f : 0.3f).SetEase(end ? Ease.InOutSine : Ease.Unset));
            await sequence.AsyncWaitForCompletion();
        }
        countdownText.gameObject.SetActive(false);
    }
    public virtual void SetStart() //�Q�[���J�n
    {
        SetAtoms(); //���q���Z�b�g
    }
    bool Check() //���w�����`�F�b�N���郁�\�b�h
    {
        bool ok = false;
        foreach (var formula in M.Formulas)
        {
            ok |= CheckConnection(formula);
        }
        return ok;
    }
    async UniTask<Vector3> PointCount() //���_���v�Z�A�\��
    {
        Vector3 center = new Vector3();
        //���_�̌v�Z
        var sortedFormulas = atomObjectHashs
            .Select(kv => kv)
            .ToList();
        sortedFormulas.Sort((a, b) => (a.Key.Point * a.Value.Count).CompareTo(b.Key.Point * b.Value.Count));
        for (int i = 0; i < sortedFormulas.Count; i++)
        {
            int prePoint = point;
            SetPoint(sortedFormulas[i].Value.Count, sortedFormulas[i].Key.Point, chainCount, comboCount); //���_���v�Z
            chainCount++; //�A�������X�V
            FormulaText(sortedFormulas[i].Key.Name).Forget();
            ChemicalText(sortedFormulas[i].Key.Formula).Forget();
            DisplayText(sortedFormulas[i].Value, prePoint).Forget();
            ScaleAtom(sortedFormulas[i].Value, out center);
            await UniTask.WaitForSeconds(1f, cancellationToken: cts.Token);
        }    
        comboCount++; //�R���{�����X�V
        chainCount = 0;
        return center;
    }
    void ScaleAtom(HashSet<Vector2Int> vector2Ints, out Vector3 center) //���������w�����g��\������
    {
        Vector3 vectorAtom = new Vector3();
        center = new Vector3();
        center.x = (vector2Ints.Max(p => p.x) + vector2Ints.Min(p => p.x)) / 2f + leftBottomPosition.x;
        center.y = (vector2Ints.Max(p => p.y) + vector2Ints.Min(p => p.y)) / 2f + leftBottomPosition.y;
        foreach (var vector2Int in vector2Ints)
        {
            AtomObject atomObject = stageAtom[vector2Int.x, vector2Int.y];
            atomObject.spriteRenderer.sortingOrder = 20;
            float scale = atomObject.transform.localScale.x;
            vectorAtom = (atomObject.transform.localPosition - center) * 0.5f + atomObject.transform.localPosition;
            var seq = DOTween.Sequence();
            seq.Append(atomObject.transform.DOScale(scale * 1.5f, 0.2f))
                .Join(atomObject.transform.DOLocalMove(vectorAtom, 0.2f))
                .AppendInterval(0.4f)
                .Append(atomObject.transform.DOScale(scale, 0.2f))
                .Join(atomObject.transform.DOLocalMove(atomObject.transform.localPosition, 0.2f))
                .OnComplete(() => atomObject.spriteRenderer.sortingOrder = 0);
        }
        center.x += transform.position.x;
        center.y += transform.position.x;
    }
    async UniTaskVoid DisplayText(HashSet<Vector2Int> vector2Ints, int prePoint)
    {
        await FormulaPointAsync(vector2Ints);
        PointAsync(prePoint);
    }
    async UniTask FormulaPointAsync(HashSet<Vector2Int> vector2Ints)//���w�����_�\��
    {
        stringBuilder.Append($"{pointSet.AtomCount.ToString()}x{pointSet.FormulaPoint.ToString()}");
        if (pointSet.ChainRate != 1) stringBuilder.Append($"x{pointSet.ChainRate.ToString()}");
        if (pointSet.ComboRate != 1) stringBuilder.Append($"x{pointSet.ComboRate.ToString()}");
        formulaPointText.text = stringBuilder.ToString(); //���_��\��
        stringBuilder.Clear();
        formulaPointText.enabled = true;
#if DEBUG
        await UniTask.WaitForSeconds(0.8f, cancellationToken: cts.Token);
#else
        vector3 = transform.position;
        vector3.x += (float)(vector2Ints.Average(p => p.x) + leftBottomPosition.x);
        vector3.y += (float)(vector2Ints.Average(p => p.y) + leftBottomPosition.y);
        formulaPointText.rectTransform.position = vector3; //���w���e�L�X�g�̈ʒu��ݒ�
        formulaPointText.rectTransform.localScale = vector3One;
        vector3.y += 0.5f;
        var seq = DOTween.Sequence();
        seq.Append(formulaPointText.rectTransform.DOMove(vector3, 0.5f).SetEase(Ease.OutCubic))
            .Join(formulaPointText.DOFade(1f, 0.1f))
            .Join(formulaPointText.rectTransform.DOScale(1.6f, 0.3f))
            .Append(formulaPointText.rectTransform.DOMove(pointText.rectTransform.position, 0.3f).SetEase(Ease.InCubic))
            .Join(DOTween.Sequence()
                .AppendInterval(0.2f)
                .Append(formulaPointText.DOFade(0f, 0.1f)));
        await seq.AsyncWaitForCompletion(); //�\������
#endif
        formulaPointText.enabled = false;
    }
    void PointAsync(int prePoint)//���_�\��
    {
        DOVirtual.Int(prePoint, point, 0.7f, value =>
        {
            pointText.text = $"{value.ToString("N0")}P";
        });
        var scaleSeq = DOTween.Sequence();
        scaleSeq.Append(pointText.rectTransform.DOScale(1.2f, 0.1f))
            .AppendInterval(0.6f)
            .Append(pointText.rectTransform.DOScale(1f, 0.1f));
    }
    async UniTask AtomDelete() //���q�̍폜
    {
        //���q�̍폜
        foreach (var atomObjectHash in atomObjectHashs)
        {
            //���������q���폜
            foreach (var atom in atomObjectHash.Value)
            {
                if (stageAtom[atom.x, atom.y] != null)
                {
                    atomObjectPool.Release(stageAtom[atom.x, atom.y]); //���q���v�[���ɖ߂�
                    stageAtom[atom.x, atom.y] = null;
                    stageAtomF[atom.x + atom.y * M.Size.x] = 0; //���q�ԍ���ݒ�
                    if (dropAtomYs[atom.x] > atom.y) dropAtomYs[atom.x] = atom.y;
                }
            }
        }
        //���������q���폜
        atomObjectHashs.Clear(); 
        await UniTask.WaitForSeconds(1.0f, ignoreTimeScale: false, cancellationToken: cts.Token); //�\������
    }
    async UniTaskVoid FormulaText(string formulaName) //���w���\��
    {
        formulaText.rectTransform.position = formulaStart;
        formulaText.enabled = true;
        formulaText.rectTransform.localScale = Vector3.one;
        formulaText.text = formulaName;
        var seq = DOTween.Sequence();
        seq.Append(formulaText.rectTransform.DOScale(1.5f, 0.6f))
            .Join(DOTween.Sequence()
                .Join(formulaText.DOFade(1f, 0.1f))
                .AppendInterval(0.4f)
                .Append(formulaText.DOFade(0f, 0.4f)));
        await seq.AsyncWaitForCompletion(); //�\������
        formulaText.enabled = false;
    }
    async UniTaskVoid ChemicalText(string formulaName) //���w���\��
    {
        chemicalText.rectTransform.position = chemicalStart;
        chemicalText.enabled = true;
        chemicalText.rectTransform.localScale = Vector3.one;
        chemicalText.text = formulaName;
        var seq = DOTween.Sequence();
        seq.Append(chemicalText.rectTransform.DOScale(1.5f, 0.6f))
            .Join(DOTween.Sequence()
                .Join(chemicalText.DOFade(1f, 0.1f))
                .AppendInterval (0.4f)
                .Append(chemicalText.DOFade(0f, 0.4f)));
        await seq.AsyncWaitForCompletion(); //�\������
        chemicalText.enabled = false;
    }
    async UniTask FullClear() //�X�e�[�W����̂Ƃ�
    {
        FormulaText("�S����").Forget();
        for (int i = 0; i < M.Size.x; i++)
        {
            AtomObject atomObject = atomObjectPool.Get();
            var scale = atomObject.transform.localScale.x;
            atomObject.Set(GetAtomColor(M.FullClearAtom), M.FullClearAtom);
            SetFreeAtom(i, 0, atomObject); //���q�̈ʒu��ݒ�
            dropAtomYs[i] = 1;
            atomObject.transform.localScale = Vector3.zero;
            atomObject.transform.DOScale(scale, 1f).SetEase(Ease.OutBounce);
        }
        await UniTask.WaitForSeconds(1.0f, ignoreTimeScale: false, cancellationToken: cts.Token); //�\������
    }
    protected bool Move(Direction direction) //�ړ����\�b�h
    {
        //�ړ��ł��邩�ǂ���
        bool isSet = false;
        //�ړ��ʂ�ݒ�
        if (direction == Direction.Right)
        {
            for (int i = 0; i < vector2Ints.Length; i++)
            {
                vector2Ints[i].x = 1;
                vector2Ints[i].y = 0;
            }
        }
        else if (direction == Direction.Left)
        {
            for (int i = 0; i < vector2Ints.Length; i++)
            {
                vector2Ints[i].x = -1;
                vector2Ints[i].y = 0;
            }

        }
        else if (direction == Direction.Down)
        {
            for (int i = 0; i < vector2Ints.Length; i++)
            {
                vector2Ints[i].x = 0;
                vector2Ints[i].y = -1;
            }
        }
        //�ړ��\�Ȃ�ړ�
        if (isSet = CheckMovement(vector2Ints))
        {
            SettingPair(vector2Ints);
        }
        return isSet;
    }
    async UniTask FreeFall(bool all = false) //���q�����R���������郁�\�b�h
    {
        List<UniTask> uniTasks = new List<UniTask>();
        if (all)
        {
            //���ׂĂ̌��q�����R����������
            for (int i = 0; i < M.Size.x; i++)
            {
                bool isBound = true;
                for (int j = dropAtomYs[i] + 1; j < M.Size.y; j++)
                {
                    if (stageAtom[i, j] != null)
                    {
                        uniTasks.Add(FreeFalling(i, j, isBound));
                        isBound = false;
                    }
                    else
                    {
                        isBound = true;
                    }
                }
            }
        }
        else
        {
            //���݂̌��q���X�e�[�W�ɐݒ�
            stageAtom[currentAtomPositions[0].x, currentAtomPositions[0].y] = currentAtoms[0];
            stageAtom[currentAtomPositions[1].x, currentAtomPositions[1].y] = currentAtoms[1];
            stageAtomF[currentAtomPositions[0].x + currentAtomPositions[0].y * M.Size.x] = (int)currentAtoms[0].AtomType + 1; //���q�ԍ���ݒ�
            stageAtomF[currentAtomPositions[1].x + currentAtomPositions[1].y * M.Size.x] = (int)currentAtoms[1].AtomType + 1; //���q�ԍ���ݒ�
            //���݂̌��q�����R����������
            int underIndex = currentAtomPositions[0].y < currentAtomPositions[1].y ? 0 : 1;
            bool isBound = currentAtomPositions[0].x != currentAtomPositions[1].x;
            uniTasks.Add(FreeFalling(currentAtomPositions[underIndex].x, currentAtomPositions[underIndex].y));
            uniTasks.Add(FreeFalling(currentAtomPositions[1 - underIndex].x, currentAtomPositions[1 - underIndex].y, isBound));
            foreach(var dropPoint in dropPoints)
            {
                dropPoint.enabled = false; //�h���b�v�n�_���\���ɂ���
            }
        }
        await UniTask.WhenAll(uniTasks);
    }
    public async void Drop() //���q���h���b�v���郁�\�b�h
    {
        if (M.CurrentState != State.Play) return;
        if (playState == PlayState.NoMove) return;
        //�h���b�v�\���ǂ���
        int dropNum = M.Size.y;
        //�h���b�v�ő�ʒu
        for(int i = 0; i < currentAtomPositions.Length; i++)
        {
            dropNum = Mathf.Min(dropNum, currentAtomPositions[i].y - dropAtomYs[currentAtomPositions[i].x]);
        }
        if (dropNum != 0)
        {
            UniTask[] uniTasks = new UniTask[currentAtoms.Length];
            var duration = dropNum * 0.06f + 0.1f;
            for (int i = 0; i < currentAtoms.Length; i++)
            {
                currentAtomPositions[i].y -= dropNum;
                var isBound = currentAtomPositions[i].y - 1 <= 0 || stageAtom[currentAtomPositions[i].x, currentAtomPositions[i].y - 1] != null;
                uniTasks[i] = DoFreeFalling(currentAtoms[i], dropNum, currentAtomPositions[i].y, isBound);
            }
            SetDropPoint(ref dropVector2Ints, ref currentAtomPositions);
            //�h���b�v�ʒu��ݒ�
            playState = PlayState.NoMove;
            await UniTask.WhenAll(uniTasks);
        }
        //��Ԃ̕ύX
        EndCheck().Forget();
    }
    bool CheckConnection(FormulaObject formula) //�Ֆʂ�AtomObject��FormulaObject�Ɣ�r���A�Ȃ���𔻒肷�郁�\�b�h
    {
        bool existed = false;

        // �Ֆʂ̃T�C�Y���擾
        int width = M.Size.x;
        int height = M.Size.y;

        //// �T���ς݃t���O
        bool[] visited = new bool[width * height];

        // �Ֆʂ𑖍�
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                if (stageAtom[x, y] != null && (!existed || (existed && !atomObjectHashs[formula].Contains(new Vector2Int(x, y)))))
                {
                    // FormulaObject�̍\���v�f���擾
                    Dictionary<AtomType, int> requiredAtoms = new Dictionary<AtomType, int>(formula.AtomDictionary);
                    checkBuffer.Clear(); // �T�������q�o�b�t�@���N���A
                    visited.AsSpan().Fill(false);
                    // �Ȃ����T��
                    if (ExploreConnections(x, y, visited, requiredAtoms, formula.AtomCount, width, height))
                    {
                        // ���ׂĂ̕K�v�Ȍ��q�����������ꍇ
                        //atomObjectHashs[formula].UnionWith(checkBuffer); // ���������q��ۑ�
                        atomObjectHashs.TryAdd(formula, new HashSet<Vector2Int>());
                        atomObjectHashs[formula].UnionWith(checkBuffer);
                        existed = true;
                    }
                }
            }
        }
        return existed; // �Ȃ��肪�������Ȃ�
    }
    bool ExploreConnections(int startX, int startY, bool[] visited, Dictionary<AtomType, int> requiredAtoms, int atomCount, int width, int height) //�K�v�Ȕ͈͓��ŗאڂ���AtomObject��T�����郁�\�b�h
    {
        Stack<Vector2Int> stack = new Stack<Vector2Int>();
        stack.Push(new Vector2Int(startX, startY));
        var vector2Int = new Vector2Int(); // �T�������q�o�b�t�@
        while (stack.Count > 0)
        {
            Vector2Int current = stack.Pop();
            int x = current.x;
            int y = current.y;
            // �T���͈͊O�̏ꍇ�͎���
            if (x < 0 || x >= width || y < 0 || y >= height) continue;

            // ���łɒT���ς݁A�܂��͋�̏ꍇ�͎���
            if (visited[GetIndex(x, y, width)] || stageAtom[x, y] == null) continue;

            // ���݂̈ʒu��T���ς݂ɐݒ�
            visited[GetIndex(x, y, width)] = true;

            // ���݂�AtomObject���擾
            AtomObject atom = stageAtom[x, y];

            // �g�p���Ȃ��܂��͕K�v���𒴂����ꍇ�͒T��������
            if (!requiredAtoms.ContainsKey(atom.AtomType) || requiredAtoms[atom.AtomType] <= 0)
            {
                continue;
            }
            // �K�v�Ȍ��q���J�E���g
            requiredAtoms[atom.AtomType]--;
            atomCount--;
            // ���������q��ۑ�
            vector2Int.x = x;
            vector2Int.y = y;
            checkBuffer.Add(vector2Int);

            // ���ׂĂ̕K�v�Ȍ��q�����������ꍇ�͏I��
            if (atomCount <= 0)
            {
                return true;
            }

            // �אڂ���ʒu���X�^�b�N�ɒǉ�
            foreach (var (dx, dy) in new[] { (1, 0), (-1, 0), (0, 1), (0, -1) })
            {
                stack.Push(new Vector2Int(x + dx, y + dy));
            }
        }
        return false; // �Ȃ��肪�������Ȃ�

        int GetIndex(int x, int y, int width) => y * width + x;
    }
    private bool CheckMovement(in Vector2Int[] position) //�ړ��̃`�F�b�N
    {
        int[] targetX = new int[position.Length];
        int[] targetY = new int[position.Length];
        bool ok = true; //�ړ��\���ǂ���
        for (int i = 0; i < targetX.Length; i++)
        {
            targetX[i] = currentAtomPositions[i].x + position[i].x;
            targetY[i] = currentAtomPositions[i].y + position[i].y;
            ok &= 0 <= targetX[i] && targetX[i] < M.Size.x && 0 <= targetY[i] && targetY[i] < M.Size.y && stageAtom[targetX[i], targetY[i]] == null;
        }
        return ok;
    }
    private void SettingPair(in Vector2Int[] positions) //���q���Z�b�g
    {
        //�ړ�
        for (int i = 0; i < currentAtomPositions.Length; i++)
        {
            currentAtomPositions[i] += positions[i];
            currentAtoms[i].transform.localPosition = currentAtomPositions[i] + leftBottomPosition; //Transform�̈ʒu��ݒ�
            SettingDisplay(currentAtomPositions[i].y, currentAtoms[i]);
        }
        SetDropPoint(ref dropVector2Ints, ref currentAtomPositions);
    }
    private void SetDropPoint(ref Vector2Int[] currents, ref Vector2Int[] currentAtomPositions) //�h���b�v�ʒu�ݒ�
    {
        if(currentAtomPositions[0].y == currentAtomPositions[1].y)
        {
            for (int i = 0; i < currentAtomPositions.Length; i++)
            {
                currents[i].x = currentAtomPositions[i].x;
                currents[i].y = dropAtomYs[currentAtomPositions[i].x];
                if (currentAtomPositions[i].y <= currents[i].y)
                {
                    dropPoints[i].enabled = false; //�h���b�v�n�_���\���ɂ���
                }
                else
                {
                    if(dropPoints[i].enabled == false) dropPoints[i].enabled = true; //�h���b�v�n�_��\���ɂ���
                    dropPoints[i].transform.localPosition = currents[i] + leftBottomPosition; //�h���b�v�n�_�̈ʒu��ݒ�
                }
            }
        }
        else
        {
            int lowerY = Mathf.Min(currentAtomPositions[0].y, currentAtomPositions[1].y);
            for (int i = 0; i < currentAtomPositions.Length; i++)
            {
                currents[i].x = currentAtomPositions[i].x;
                currents[i].y = dropAtomYs[currentAtomPositions[i].x] + i;
                if (lowerY <= currents[i].y)
                {
                    dropPoints[i].enabled = false; //�h���b�v�n�_���\���ɂ���
                }
                else
                {
                    if (dropPoints[i].enabled == false) dropPoints[i].enabled = true; //�h���b�v�n�_��\���ɂ���
                    dropPoints[i].transform.localPosition = currents[i] + leftBottomPosition; //�h���b�v�n�_�̈ʒu��ݒ�
                }
            }
        }
    }
    private void SettingDisplay(int y, AtomObject atomObject) //���q���Z�b�g
    {
        if (y >= displayUpper)
        {
            atomObject.UnEnabled();
        }
        else
        {
            atomObject.Enabled();
        }
    }
    protected void Rotation(Direction direction) //���q����]�����郁�\�b�h
    {
        if (M.CurrentState != MainManager.State.Play) return;
        //��]�\���ǂ���
        bool isSet = false;
        //��]�����擾
        RotationInfo rotationInfo = RotationInfo.Top;
        if (currentAtomPositions[0].x < currentAtomPositions[1].x) rotationInfo = RotationInfo.Right;
        else if (currentAtomPositions[0].x > currentAtomPositions[1].x) rotationInfo = RotationInfo.Left;
        else if (currentAtomPositions[0].y > currentAtomPositions[1].y) rotationInfo = RotationInfo.Down;
        //��]�ʒu��ݒ�
        SetRotationPosition(ref vector2Ints, direction, rotationInfo);
        //��]�\�Ȃ��]
        if (isSet = CheckMovement(vector2Ints))
        {
            SettingPair(vector2Ints);
        }
        else
        {
            SetRotationPosition2(ref vector2Ints, direction, rotationInfo);
            if (isSet = CheckMovement(vector2Ints))
            {
                SettingPair(vector2Ints);
            }
        }
    }
    void SetRotationPosition(ref Vector2Int[] vector2Ints, Direction direction, RotationInfo rotationInfo) //��]�ʒu���Z�b�g
    {
        //���͂܂��Ȃ�
        vector2Ints[0].x = 0;
        vector2Ints[0].y = 0;
        //��]�ʒu��ݒ�
        if (direction == Direction.Left)
        {
            if (rotationInfo == RotationInfo.Top)
            {
                vector2Ints[1].x = 1;
                vector2Ints[1].y = -1;
            }
            else if (rotationInfo == RotationInfo.Right)
            {
                vector2Ints[1].x = -1;
                vector2Ints[1].y = -1;
            }
            else if (rotationInfo == RotationInfo.Down)
            {
                vector2Ints[1].x = -1;
                vector2Ints[1].y = 1;
            }
            else
            {
                vector2Ints[1].x = 1;
                vector2Ints[1].y = 1;
            }
        }
        else if (direction == Direction.Right)
        {
            if (rotationInfo == RotationInfo.Top)
            {
                vector2Ints[1].x = -1;
                vector2Ints[1].y = -1;
            }
            else if (rotationInfo == RotationInfo.Right)
            {
                vector2Ints[1].x = -1;
                vector2Ints[1].y = 1;
            }
            else if (rotationInfo == RotationInfo.Down)
            {
                vector2Ints[1].x = 1;
                vector2Ints[1].y = 1;
            }
            else
            {
                vector2Ints[1].x = 1;
                vector2Ints[1].y = -1;
            }
        }
    }
    void SetRotationPosition2(ref Vector2Int[] vector2Ints, Direction direction, RotationInfo rotationInfo) //��]�ʒu���Z�b�g
    {
        //��]�ʒu��ݒ�
        if (direction == Direction.Left)
        {
            if (rotationInfo == RotationInfo.Top)
            {
                vector2Ints[0].x -= 1;
                vector2Ints[1].x -= 1;
            }
            else if (rotationInfo == RotationInfo.Right)
            {
                vector2Ints[0].y += 1;
                vector2Ints[1].y += 1;
            }
            else if (rotationInfo == RotationInfo.Down)
            {
                vector2Ints[0].x += 1;
                vector2Ints[1].x += 1;
            }
        }
        else if (direction == Direction.Right)
        {
            if (rotationInfo == RotationInfo.Top)
            {
                vector2Ints[0].x += 1;
                vector2Ints[1].x += 1;
            }
            else if (rotationInfo == RotationInfo.Down)
            {
                vector2Ints[0].x -= 1;
                vector2Ints[1].x -= 1;
            }
            else if (rotationInfo == RotationInfo.Left)
            {
                vector2Ints[0].y += 1;
                vector2Ints[1].y += 1;
            }
        }
    }
    async UniTask FreeFalling(int x, int y, bool isBound = true) //���q�����R���������郁�\�b�h
    {
        if (dropAtomYs[x] < y)
        {
            //�����ʒu
            AtomObject atomObject = stageAtom[x, y]; //���q���擾
            stageAtom[x, dropAtomYs[x]] = atomObject; //���̌��q��ۑ�
            stageAtomF[x + dropAtomYs[x] * M.Size.x] = (int)atomObject.AtomType + 1; //���q�ԍ���ݒ�
            stageAtom[x, y] = null; //���̈ʒu��null�ɂ���
            stageAtomF[x + y * M.Size.x] = 0; //���q�ԍ���ݒ�
            var height = y - dropAtomYs[x];
            dropAtomYs[x]++;
            //���q���ړ�
            await DoFreeFalling(atomObject, height, dropAtomYs[x] - 1, isBound);
        } 
        else
        {
            dropAtomYs[x]++;
            SettingDisplay(y, stageAtom[x, y]);
        }
    }
    async UniTask DoFreeFalling(AtomObject atomObject, int height, int y, bool isBound)
    {
        var tween = atomObject.transform.DOLocalMoveY(y + leftBottomPosition.y, height * 0.02f)
            .SetEase(Ease.Linear)
            .OnUpdate(() => 
            {
                if (atomObject.transform.localPosition.y >= displayUpper + leftBottomPosition.y)
                {
                    atomObject.UnEnabled();
                }
                else
                {
                    atomObject.Enabled();
                }
            });
        await tween.AsyncWaitForCompletion(); //�ړ�����
        var scale = atomObject.transform.localScale;
        scale.y *= isBound ? 0.87f : 0.91f;
        scale.x *= isBound ? 1.13f : 1.09f;
        tween = atomObject.transform.DOScale(scale, 0.15f)
            .SetLoops(2, LoopType.Yoyo)
            .SetEase(Ease.InOutSine);
        await tween.AsyncWaitForCompletion();
    }
    int DestroyGotDisturbanceNumber(int num) //����܌��q���폜���郁�\�b�h
    {
        Vector3 vector3 = disturbancAtomPosition;
        var pass = num - gotDisturbanceNumber;
        var start = gotDisturbanceNumber;
        gotDisturbanceNumber = Mathf.Max(gotDisturbanceNumber - num, 0);
        for (int i = start - 1; i >= gotDisturbanceNumber; i--)
        {
            M.DisturbanceAtomObjects[i].UnEnabled(); //����܌��q���폜
            M.DisturbanceAtomObjects[i].transform.parent = M.StockAtoms.transform;
        }
        if(start <= NormalMax) return pass;
        if (NormalMax < gotDisturbanceNumber)
        {
            for (int i = 1; i < gotDisturbanceNumber; i++)
            {
                vector3.x += (M.DisturbancAtomsSize * (NormalMax - 1)) / (gotDisturbanceNumber - 1);
                M.DisturbanceAtomObjects[i].transform.DOLocalMove(vector3, 0.5f);
            }
        }
        else
        {
            for (int i = 1; i < gotDisturbanceNumber; i++)
            {
                vector3.x += M.DisturbancAtomsSize;
                M.DisturbanceAtomObjects[i].transform.DOLocalMove(vector3, 0.5f);
            }
        }
        return pass;
    }
    bool DropGotDisturbanceNumber() //����܌��q�𗎂Ƃ����\�b�h
    {
        if (gotDisturbanceNumber <= 0)
        {
            return false;
        }
        int width = M.Size.x;
        int height = M.Size.y;
        int count = 0;
        bool[] isSet = new bool[width];
        for(int i = 0; i < gotDisturbanceNumber; i++)
        {
            M.DisturbanceAtomObjects[i].UnEnabled();
            M.DisturbanceAtomObjects[i].transform.parent = M.StockAtoms.transform;
        }
        for (int y = height - 1; y >= 0; y--)
        {
            for (int i = 0; i < width; i++)
            {
                var x = (disturbancStartX + i) % width;
                if (isSet[x]) continue;
                if (stageAtom[x, y] == null)
                {
                    var disturbanceAtomObject = atomObjectPool.Get();
                    disturbanceAtomObject.Set(M.DisturbanceAtomColor, M.DisturbanceAtom);
                    SetFreeAtom(x, y, disturbanceAtomObject); //���q�̈ʒu��ݒ�
                    count++;
                    if(count >= gotDisturbanceNumber)
                    {
                        disturbancStartX = (x + 1) % width;
                        gotDisturbanceNumber = 0;
                        return true;
                    }
                }
                else
                {
                    isSet[x] = true; //���܂��Ă���
                    //���ׂĖ��܂��Ă�����
                    var all = isSet.All(p => p);
                    if (all)
                    {
                        disturbancStartX = (x + 1) % width;
                        gotDisturbanceNumber = 0;
                        return true;
                    }
                }
            }
        }
        gotDisturbanceNumber = 0;
        return true;
    }
    void SetDisturbance(Vector3 center) //������܌��q���Z�b�g���郁�\�b�h
    {
        var disturbanceCount = point / 400 - passTotalDisturbanceNumber; //�ǉ�����܌��q
        passTotalDisturbanceNumber = point / 400; //�X�V
        if (disturbanceCount > 0)
        {
            var disturbanceNumber = DestroyGotDisturbanceNumber(disturbanceCount); //����܌��q���폜
            if (disturbanceNumber > 0) M.PassDisturbance(disturbanceNumber, this, center); //����܌��q�̐���n��
        }
    }
    private void OnDestroy() //�폜���A�^�X�N���L�����Z��
    {
        //�L�����Z���g�[�N�����L�����Z��
        if (cts != null)
        {
            cts.Cancel();
            cts.Dispose();
            cts = null;
        }
    }
#endregion ����J���\�b�h
}
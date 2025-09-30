using UnityEngine;
using UnityEngine.InputSystem;
using System.Linq;
//��������'***'�ŕ\�L

public class PlayerManager : PlayerBase
{
    [SerializeField] InputActionAsset inputActions; //���̓A�N�V�����A�Z�b�g
    [SerializeField] float moveSensitivity; //���͂̊��x
    Vector2 moveInput; //�ړ�����
    float rotateInput; //��]����
    float continuousRightCount; //�A���E�J�E���g
    float continuousLeftCount; //�A�����J�E���g
    float continuousRightRotateCount; //�A���E��]�J�E���g
    float continuousLeftRotateCount; //�A������]�J�E���g
    public override void Start0(int id) //�v���C���[�̏��������s�����\�b�h
    {
        base.Start0(id);
        //����̏�����
        SetInputEvent();
    }
    private void SetInputEvent() //���̓C�x���g��ݒ肷�郁�\�b�h
    {
        var actionMaps = inputActions.actionMaps.ToDictionary(x => x.name, x => x);
        actionMaps.TryGetValue("Player", out var playerMap);
        var actionPlayer = playerMap.ToDictionary(x => x.name, x => x);
        actionPlayer.TryGetValue("Move", out var move);
        actionPlayer.TryGetValue("Drop", out var drop);
        actionPlayer.TryGetValue("Rotation", out var rotation);
        actionPlayer.TryGetValue("Pause", out var pause);
        move.performed += context => moveInput = context.ReadValue<Vector2>();
        move.canceled += context => moveInput = context.ReadValue<Vector2>();
        drop.started += context => Drop();
        rotation.performed += context => rotateInput = context.ReadValue<float>();
        rotation.canceled += context => rotateInput = context.ReadValue<float>();
        pause.performed += context => M.Pause();
    }
    protected override void PlayMove() //�ړ����͂��������郁�\�b�h
    {
        //��]����
        if (rotateInput > 0.5f)
        {
            //�A���E��]�J�E���g���X�V
            continuousRightRotateCount += Time.deltaTime;
            //�A���E�J�E���g����莞�Ԃ𒴂�����E�ړ�
            if (continuousRightRotateCount >= M.ContinuousMoveTime)
            {
                Rotation(Direction.Right);
                continuousRightRotateCount = 0; //�A���E�J�E���g�����Z�b�g
            }
        }
        else
        {
            //�A���E��]�J�E���g�����Z�b�g
            continuousRightRotateCount = M.ContinuousMoveTime;
        }
        if (rotateInput < -0.5f)
        {
            //�A������]�J�E���g���X�V
            continuousLeftRotateCount += Time.deltaTime;
            //�A������]�J�E���g����莞�Ԃ𒴂����獶�ړ�
            if (continuousLeftRotateCount >= M.ContinuousMoveTime)
            {
                Rotation(Direction.Left);
                continuousLeftRotateCount = 0; //�A�����J�E���g�����Z�b�g
            }
        }
        else
        {
            //�A������]�J�E���g�����Z�b�g
            continuousLeftRotateCount = M.ContinuousMoveTime;
        }
        //�ړ�����
        if (moveInput.x > moveSensitivity)
        {
            //�A���E�J�E���g���X�V
            continuousRightCount += Time.deltaTime;
            //�A���E�J�E���g����莞�Ԃ𒴂�����E�ړ�
            if (continuousRightCount >= M.ContinuousMoveTime)
            {
                Move(Direction.Right);//, currentAtomPositions, stageAtom); //���q���E�Ɉړ�
                continuousRightCount = 0; //�A���E�J�E���g�����Z�b�g
            }
        }
        else
        {
            //�A���E�J�E���g�����Z�b�g
            continuousRightCount = M.ContinuousMoveTime;
        }
        if (moveInput.x < -moveSensitivity)
        {
            //�A�����J�E���g���X�V
            continuousLeftCount += Time.deltaTime;
            //�A�����J�E���g����莞�Ԃ𒴂����獶�ړ�
            if (continuousLeftCount >= M.ContinuousMoveTime)
            {
                Move(Direction.Left);//, currentAtomPositions, stageAtom); //���q�����Ɉړ�
                continuousLeftCount = 0; //�A�����J�E���g�����Z�b�g
            }
        }
        else
        {
            //�A�����J�E���g�����Z�b�g
            continuousLeftCount = M.ContinuousMoveTime;
        }
        float dropValue = 0;
        if (moveInput.y < -moveSensitivity)
        {
            //�h���b�v�J�E���g���X�V
            dropValue = M.DownAcceleration * Time.deltaTime;
        }
        else
        {
            //�h���b�v�J�E���g���X�V
            dropValue = Time.deltaTime;
        }
        DropStep(dropValue); //�h���b�v����
    }
}


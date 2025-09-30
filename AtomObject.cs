using TMPro;
using Unity.VisualScripting;
using UnityEngine;

public class AtomObject : MonoBehaviour
{
    public SpriteRenderer spriteRenderer;
    public MainManager.AtomType AtomType;
    public TextMeshPro text;
    public void Set(Color color , MainManager.AtomType AtomType)
    {
        spriteRenderer.color = color;
        this.AtomType = AtomType;
        text.text = AtomType.ToString();
        text.color = color;
    }
    public void Enabled()
    {
        if (!spriteRenderer.enabled)
        {
            spriteRenderer.enabled = true;
            text.enabled = true;
        }
    }
    public void UnEnabled()
    {
        if (spriteRenderer.enabled)
        {
            spriteRenderer.enabled = false;
            text.enabled = false;
        }
    }
}

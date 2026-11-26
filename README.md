# Architecture_Transformer

Implémentation et étude de l'architecture Transformer (self-attention, multi-head attention, encodeur-décodeur) - TP EMSI 2026/2027

## Contenu du dépôt

| Fichier | Description |
|---|---|
| `Architecture_transformer.ipynb` | Notebook principal : implémentation, entraînement et évaluation des modèles |
| `mini_bert.pt` | Poids du modèle Mini-BERT (encodeur seul) |
| `mini_gpt.pt` | Poids du modèle Mini-GPT (décodeur seul) |
| `energy_transformer.pt` | Poids du Transformer entraîné sur les données d'énergie |
| `transformer_168.pt` | Poids du Transformer avec une fenêtre de 168 pas de temps |

## Concepts abordés

- Mécanisme de self-attention
- Multi-head attention
- Encodage positionnel
- Architecture encodeur-décodeur
- Variantes : encodeur seul (type BERT) et décodeur seul (type GPT)

## Prérequis

- Python 3.9 ou plus
- PyTorch
- NumPy, Pandas, Matplotlib
- Jupyter Notebook ou JupyterLab

```bash
pip install torch numpy pandas matplotlib jupyter
```

## Utilisation

1. Cloner le dépôt :

```bash
git clone https://github.com/malaklotfi911-source/Architecture_Transformer.git
cd Architecture_Transformer
```

2. Lancer le notebook :

```bash
jupyter notebook Architecture_transformer.ipynb
```

3. Pour charger un modèle déjà entraîné :

```python
import torch

state_dict = torch.load("mini_gpt.pt", map_location="cpu")
# model.load_state_dict(state_dict)
```

## Auteure

LOTFI Malak - 5IIR, EMSI Roudani

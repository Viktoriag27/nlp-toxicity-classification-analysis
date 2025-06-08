import pandas as pd
import numpy as np

def overall_results():
    overall_results = [
    {"pct":  1,  "acc": 0.9052, "prec": 0.8636, "rec": 0.8447, "f1": 0.8536},
    {"pct": 10,  "acc": 0.9275, "prec": 0.8922, "rec": 0.8888, "f1": 0.8905},
    {"pct": 25,  "acc": 0.9325, "prec": 0.8883, "rec": 0.9190, "f1": 0.9023},
    {"pct": 50,  "acc": 0.9499, "prec": 0.9171, "rec": 0.9364, "f1": 0.9263},
    {"pct": 75,  "acc": 0.9639, "prec": 0.9389, "rec": 0.9551, "f1": 0.9467},
    {"pct":100,  "acc": 0.9829, "prec": 0.9731, "rec": 0.9755, "f1": 0.9743},
    ]
    df_metrics = pd.DataFrame(overall_results).sort_values("pct")
    return df_metrics


def perlabel_results():
    perlabel_results = [
    # 1%
    {"pct":1, "label":"toxic", "f1":0.8770},
    {"pct":1, "label":"severe_toxic", "f1":0.0000},
    {"pct":1, "label":"obscene", "f1":0.7761},
    {"pct":1, "label":"insult", "f1":0.6881},
    {"pct":1, "label":"identity_hate", "f1":0.0000},

    # 10%
    {"pct":10, "label":"toxic", "f1":0.9015},
    {"pct":10, "label":"severe_toxic", "f1":0.5328},
    {"pct":10, "label":"obscene", "f1":0.8402},
    {"pct":10, "label":"insult", "f1":0.7556},
    {"pct":10, "label":"identity_hate", "f1":0.5832},

    # 25%
    {"pct":25, "label":"toxic", "f1":0.9165},
    {"pct":25, "label":"severe_toxic", "f1":0.5773},
    {"pct":25, "label":"obscene", "f1":0.8647},
    {"pct":25, "label":"insult", "f1":0.7915},
    {"pct":25, "label":"identity_hate", "f1":0.6618},

    # 50%
    {"pct":50, "label":"toxic", "f1":0.9379},
    {"pct":50, "label":"severe_toxic", "f1":0.6569},
    {"pct":50, "label":"obscene", "f1":0.8984},
    {"pct":50, "label":"insult", "f1":0.8408},
    {"pct":50, "label":"identity_hate", "f1":0.7283},

    # 75%
    {"pct":75, "label":"toxic", "f1":0.9537},
    {"pct":75, "label":"severe_toxic", "f1":0.7460},
    {"pct":75, "label":"obscene", "f1":0.9286},
    {"pct":75, "label":"insult", "f1":0.8883},
    {"pct":75, "label":"identity_hate", "f1":0.7964},

    # 100%
    {"pct":100, "label":"toxic", "f1":0.9818},
    {"pct":100, "label":"severe_toxic", "f1":0.8397},
    {"pct":100, "label":"obscene", "f1":0.9660},
    {"pct":100, "label":"insult", "f1":0.9428},
    {"pct":100, "label":"identity_hate", "f1":0.9084},
    ]

    df = pd.DataFrame(perlabel_results)
    return df


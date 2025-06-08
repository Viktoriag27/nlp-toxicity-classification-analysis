import time
import numpy as np
import pandas as pd
from transformers import (
    MarianMTModel, MarianTokenizer, pipeline
)
from datasets import Dataset
import torch
from tqdm import tqdm
import google.generativeai as genai

class DataAugmentor:
    """
    Encapsulates Part 2 techniques (back-translation, zero-shot via HF pipeline, synthetic generation).
    """
    def __init__(self, label_cols: list[str]):
        self.label_cols = label_cols
        
        # MarianMT models for back-translation
        self.tokenizer_en_it = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-it")
        self.model_en_it = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-it")
        self.tokenizer_it_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-it-en")
        self.model_it_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-it-en")

        # The HF zero-shot pipeline (Bart-MNLI by default)
        self.zero_shot_classifier = pipeline("zero-shot-classification")

        #  Gemini API for synthetic generation
        self.gemini_api_key = "AIzaSyCulx43Z3mDniEih9JZ3FLCa2wQVAWJFUc"

    def backtranslate(self, df_gold: pd.DataFrame) -> pd.DataFrame:
        """
        1) For each row in df_gold , back-translate comment_text (en→it→en).
        2) Skip very short texts (<10 chars). 
        3) Collect only those back-translations that are non-null, differ from the original, and length>10.
        4) Build augmented_df with new texts + copied labels.
        5) Concatenate augmented_df with df_gold → combined_df.
        6) Tokenize combined_df and fine-tune DistilBERT via _train_and_eval.
        7) Return validation metrics.
        """

        # 0) Helper function to back-translate a single text
        def back_translate_text(text: str) -> str | None:
            text = text.strip()
            if len(text) < 10:
                return None
            try:
                # English → Italian
                en_inputs = self.tokenizer_en_it(
                    text, return_tensors="pt", padding=True, truncation=True, max_length=128
                )
                it_tokens = self.model_en_it.generate(
                    **en_inputs, max_length=128, num_beams=4, early_stopping=True
                )
                italian_text = self.tokenizer_en_it.decode(it_tokens[0], skip_special_tokens=True)

                # Italian → English
                it_inputs = self.tokenizer_it_en(
                    italian_text, return_tensors="pt", padding=True, truncation=True, max_length=128
                )
                back_tokens = self.model_it_en.generate(
                    **it_inputs, max_length=128, num_beams=4, early_stopping=True
                )
                final_text = self.tokenizer_it_en.decode(back_tokens[0], skip_special_tokens=True)
                return final_text.strip()
            except Exception as e:
                print(f"Error translating: {str(e)[:50]}...")
                return None

        # 1) Iterate over df_gold rows and collect back-translations + labels
        translated_texts = []
        translated_labels = []
        for idx, row in tqdm(df_gold.iterrows(), total=len(df_gold), desc="Back-translating"):
            original_text = row["comment_text"]
            if len(original_text.strip()) < 10:
                continue

            translated = back_translate_text(original_text)
            if translated and translated != original_text and len(translated) > 10:
                translated_texts.append(translated)
                # Copy the original five labels verbatim
                label_row = {lbl: int(row[lbl]) for lbl in self.label_cols}
                translated_labels.append(label_row)

        # 2) Build augmented_df DataFrame
        augmented_data = []
        for text, labels in zip(translated_texts, translated_labels):
            row_dict = {"comment_text": text}
            row_dict.update(labels)
            augmented_data.append(row_dict)
        augmented_df = pd.DataFrame(augmented_data)

        print(f"Original examples: {len(df_gold)}")
        print(f"Back-translated examples: {len(augmented_df)}")

        # 3) Combine df_gold + augmented_df
        combined_df = pd.concat([df_gold.reset_index(drop=True), augmented_df.reset_index(drop=True)], ignore_index=True)

        return combined_df
    

    def run_synthetic(self, df_gold: pd.DataFrame, n_gen: int) -> pd.DataFrame:
        """
        Generate 'n_gen' synthetic toxic comments via Gemini and concatenate with 'df_gold'.
        """
        # Configure API (ideally done once in __init__)
        genai.configure(api_key=self.gemini_api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-lite')

        

        def _generate_batch(batch_size: int) -> list[dict]:
            prompt = f"""
Generate {batch_size} realistic toxic online comments. Each comment should be labeled with the toxicity categories it belongs to.

Categories:
- toxic: Rude, disrespectful language
- severe_toxic: Very hateful, aggressive language
- obscene: Swear words, curse words
- insult: Insulting language toward a person/group
- identity_hate: Hateful language targeting identity (race, gender, religion)

For each comment, provide:
COMMENT: [text]
LABELS: toxic=1, severe_toxic=0, obscene=1, insult=0, identity_hate=0
---
"""
            response = model.generate_content(prompt)
            text = response.text.strip()
            examples = []
            for section in text.split('---'):
                if 'COMMENT:' in section and 'LABELS:' in section:
                    lines = section.strip().splitlines()
                    comment, labels = '', {}
                    for col in self.label_cols:
                        labels[col] = 0
                    for line in lines:
                        if line.startswith('COMMENT:'):
                            comment = line.split('COMMENT:')[1].strip()
                        elif line.startswith('LABELS:'):
                            for pair in line.split('LABELS:')[1].split(','):
                                key, val = pair.strip().split('=')
                                if key in labels and val in ('0','1'):
                                    labels[key] = int(val)
                    if comment and len(comment.split()) >= 3:
                        examples.append({'comment_text': comment, **labels})
            return examples[:batch_size]

        # Generate until we have n_gen examples, respecting API rate limits
        synthetic = []
        batch_size = min(30, n_gen)
        while len(synthetic) < n_gen:
            needed = n_gen - len(synthetic)
            synthetic.extend(_generate_batch(min(batch_size, needed)))
            if len(synthetic) < n_gen:
                time.sleep(65)
        synthetic_df = pd.DataFrame(synthetic[:n_gen])

        # Combine with gold set
        combined = pd.concat([df_gold.reset_index(drop=True), synthetic_df.reset_index(drop=True)], ignore_index=True)
        return combined



    def zero_shot_pipeline(
    self,
    df_gold: pd.DataFrame,
    df_unlabeled: pd.DataFrame,
    confidence_threshold: float = 0.6,
    batch_size: int = 16
    ) -> pd.DataFrame:
        """
        Efficient version using GPU-accelerated batched inference via HF pipeline.
        """
        classifier = self.zero_shot_classifier
        label_cols = self.label_cols

        texts = df_unlabeled["comment_text"].tolist()
        new_rows = []

        print(f"Running zero-shot classification on {len(texts)} examples (batch_size={batch_size})")
        for i in tqdm(range(0, len(texts), batch_size), desc="Zero-shot classification"):
            batch_texts = texts[i: i + batch_size]

            try:
                batch_results = classifier(batch_texts, candidate_labels=label_cols)
            except Exception as e:
                print(f"Batch error: {str(e)}")
                continue

            # batch_results is a list of dicts (one per text)
            for j, result in enumerate(batch_results):
                top_label = result["labels"][0]
                top_score = result["scores"][0]

                if top_score >= confidence_threshold:
                    row = {"comment_text": batch_texts[j]}
                    for col in label_cols:
                        row[col] = 1 if col == top_label else 0
                    new_rows.append(row)

        augmented_df = pd.DataFrame(new_rows)
        print(f"\nZero-shot pipeline kept {len(augmented_df)} examples (threshold={confidence_threshold})")

        combined_df = pd.concat(
            [df_gold.reset_index(drop=True), augmented_df.reset_index(drop=True)],
            ignore_index=True
        )
        return combined_df




# /gen-data

Generate synthetic fine-tuning examples for the LoRA. Examples must cover:
- Correct brief format (all six sections, structured headers)
- Deflection behaviour on symptom/dose/diagnosis queries
- OCR reading extraction and logging

Steps:
1. Run `python finetune/generate_data.py --count 200 --out finetune/data/train.jsonl`
2. Run `python finetune/generate_data.py --count 50 --out finetune/data/eval.jsonl --seed 99`
3. Validate the output: check JSONL is valid, each example has `prompt` and `completion` keys, no example contains a diagnostic conclusion or dose advice
4. Print counts: total examples, brief-format examples, deflection examples, OCR examples
5. If validation fails, show the offending examples

Target: 150-400 total training examples reviewed as clinically plausible before training.

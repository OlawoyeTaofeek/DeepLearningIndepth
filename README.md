# DeepLearning-Indepth

- Deep Learning Foundation
- CNN, RNN and LSTM
- Preparing unstructured Data for modeling
- Transformer Architecture and Evolution
- Build a Large Language model from Scratch
- Pydantic for LLM output Formating
- HuggingFace
- 4 project with deployment on CNN, RNN, and LLM
- Agents and RAG
- LangChain
- LangGraph
- LangSmith
- OpenAI
- CrewAI
- AnthopicAI
- LlamaIndex
- Ollama
- Gemini
- MCPs
- Agentic AI
- FastAPI for deployment
- Experiment Tracking: MLflow, Evidently, Grafana
- Containerization and Virtualization: Docker and Kubernetes
- AWS Cloud; Deployment with AWS Cloud, Amazon Sagemaker 


```python
#!/usr/bin/env python3
"""
train_text_classification.py

Fine-tune a transformer for single-label text classification with Hugging Face Transformers + Datasets.
Supports: automatic dataset column detection, tokenization, Trainer training loop, evaluation, saving, push-to-hub.

Example:
  python train_text_classification.py \
    --model_name_or_path distilbert-base-uncased \
    --dataset glue --dataset_config_name sst2 \
    --output_dir ./outputs/distilbert-sst2 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 64 \
    --learning_rate 2e-5 \
    --push_to_hub
"""

import argparse
import logging
import os
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import evaluate
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune transformer for text classification")
    parser.add_argument("--model_name_or_path", type=str, default="distilbert-base-uncased")
    parser.add_argument("--dataset", type=str, default="glue", help="Dataset identifier or local path")
    parser.add_argument("--dataset_config_name", type=str, default="sst2", help="Dataset config (e.g., sst2 for glue)")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Optional: repo name on HF Hub")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision if available")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint or 'auto'")
    return parser.parse_args()


def find_text_and_label_columns(dataset) -> (str, str):
    """
    Heuristic to find which columns correspond to text and label.
    Returns (text_column, label_column)
    """
    sample = dataset["train"][0]
    # candidate text columns
    text_candidates = ["text", "sentence", "review", "article", "content"]
    label_candidates = ["label", "labels", "stars"]

    text_column = None
    label_column = None
    for c in text_candidates:
        if c in sample:
            text_column = c
            break
    if text_column is None:
        # fallback: choose first string column
        for k, v in sample.items():
            if isinstance(v, str):
                text_column = k
                break
    for c in label_candidates:
        if c in sample:
            label_column = c
            break
    if label_column is None:
        # fallback: first int column
        for k, v in sample.items():
            if isinstance(v, int):
                label_column = k
                break
    if text_column is None or label_column is None:
        raise ValueError(f"Couldn't infer text/label columns. Sample keys: {list(sample.keys())}")
    return text_column, label_column


def preprocess_function(examples, tokenizer, text_column: str, max_length: int):
    texts = examples[text_column]
    # tokenizer can handle batch input
    return tokenizer(texts, truncation=True, max_length=max_length)


def compute_metrics_fn(eval_pred) -> Dict[str, float]:
    # For single-label classification
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = metric_acc.compute(predictions=preds, references=labels)
    f1 = metric_f1.compute(predictions=preds, references=labels, average="macro")
    return {"accuracy": acc["accuracy"], "f1_macro": f1["f1"]}


def main():
    args = parse_args()
    set_seed(args.seed)

    logger.info("Loading dataset %s (%s)", args.dataset, args.dataset_config_name)
    raw_datasets = load_dataset(args.dataset, args.dataset_config_name) if args.dataset_config_name else load_dataset(args.dataset)

    text_column, label_column = find_text_and_label_columns(raw_datasets)
    logger.info("Identified text column: %s, label column: %s", text_column, label_column)

    # Build label mapping
    # If labels are ints starting at 0, we can use num_labels directly
    # Otherwise, we build a list of unique labels from train
    unique_labels = sorted(set(raw_datasets["train"][label_column])) if "train" in raw_datasets else sorted(set(raw_datasets[list(raw_datasets.keys())[0]][label_column]))
    num_labels = len(unique_labels)
    logger.info("Number of labels: %d", num_labels)

    # Load tokenizer & model
    logger.info("Loading tokenizer and model from %s", args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)

    # If tokenizer has more tokens than model embeddings, resize embeddings
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        logger.info("Resizing model embeddings to %d tokens", len(tokenizer))
        model.resize_token_embeddings(len(tokenizer))

    # Tokenize datasets
    logger.info("Tokenizing datasets (text col = %s)", text_column)
    tokenized_datasets = raw_datasets.map(
        lambda examples: preprocess_function(examples, tokenizer, text_column=text_column, max_length=args.max_length),
        batched=True,
        remove_columns=[col for col in raw_datasets["train"].column_names if col not in [text_column, label_column]],
    )

    # Rename label column to "labels" expected by Trainer
    tokenized_datasets = tokenized_datasets.map(lambda examples: {"labels": examples[label_column]}, batched=True, remove_columns=[label_column])

    train_dataset = tokenized_datasets["train"] if "train" in tokenized_datasets else tokenized_datasets[list(tokenized_datasets.keys())[0]]
    eval_dataset = tokenized_datasets["validation"] if "validation" in tokenized_datasets else (tokenized_datasets.get("test", None))

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        logging_steps=args.logging_steps,
        logging_dir=os.path.join(args.output_dir, "logs"),
        seed=args.seed,
        load_best_model_at_end=True if args.evaluation_strategy != "no" else False,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=args.fp16 and torch.cuda.is_available(),
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn if eval_dataset is not None else None,
    )

    # Resume training if requested
    resumed = False
    if args.resume_from_checkpoint:
        ckpt = args.resume_from_checkpoint
        if ckpt == "auto":
            ckpt = trainer._load_from_checkpoint()  # may be None
        if ckpt:
            logger.info("Resuming training from checkpoint: %s", ckpt)
            trainer.train(resume_from_checkpoint=ckpt)
            resumed = True

    if not resumed:
        logger.info("Starting training")
        trainer.train()

    logger.info("Saving final model to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if eval_dataset is not None:
        logger.info("Running final evaluation")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        logger.info("Evaluation results: %s", metrics)

    if args.push_to_hub:
        logger.info("Pushing model to the Hugging Face Hub")
        trainer.push_to_hub(**({"commit_message": "Push model via training script"} if args.hub_model_id else {}))

    logger.info("Done.")


if __name__ == "__main__":
    main()
```


So, later, you will maintain what we have, but add login, signup, a way to switch between been a vendor, and, and a customer or what do you think, can a service company also be a customers or how is it going to be like, like I want the login, signup to be well structure to consider them perfectly, also the name will change to AfriConnect, connecting those in need to those who have or can offer...... Also as a vendor remeber he or she should be able to track inventory, orders, delivery, analytics for analytics, bulk notification access for reminder, newly added products, and more, and also a way to always get month end analytics of the whole month, tracking clicks, orders, completed transactions, sales amount made and all. same with the service providers, they should be able to have analysis and more if their own. Lastly, when I click on a vendor, I should be able to see what they sell, order, add to carts, and also maybe, order at once or click chat to have conversation with the vendor...


# 30 Days of Serious Python — Days 1–3 (Detailed)

Each day follows this structure: **Concept explanation → Annotated code examples → Common mistakes → Project (with full solution) → Practice exercises.**

Work top to bottom. Try the project yourself before reading the solution.

---

# DAY 1 — Syntax, Variables, Types, Operators

## 1. Concept Explanation

### Variables
Python variables are **names bound to objects** — not boxes holding values. `x = 5` means "the name `x` now points to an integer object `5`", not "copy 5 into a memory slot called x."

```python
x = 5
y = x       # y points to the same object as x
x = 10      # x now points to a NEW object; y is untouched
print(y)    # 5
```

### Core types
```python
age = 25             # int
price = 19.99          # float
name = "Alice"          # str
is_active = True          # bool
nothing = None              # NoneType — represents "no value"

print(type(age))      # <class 'int'>
```

### Type conversion (casting)age = "45
```python
str(25)        # "25"
int("25")      # 25
float("3.14")   # 3.14
int(3.9)        # 3  (truncates, doesn't round!)
bool(0)          # False
bool("")          # False
bool("hello")      # True — any non-empty string is truthy
```
**Gotcha:** `int("3.14")` raises `ValueError` — go through `float` first: `int(float("3.14"))`.

### Arithmetic operators
```python
7 + 3    # 10
7 - 3    # 4
7 * 3    # 21
7 / 3    # 2.333...  (true division — always returns float)
7 // 3   # 2          (floor division)
7 % 3    # 1          (modulo — remainder)
7 ** 2   # 49         (exponent)
```

### Comparison & logical operators
```python
5 == 5     # True  (equality)
5 != 4     # True  (inequality)
5 > 3 and 2 < 4    # True
5 > 3 or 1 > 4     # True
not True             # False
```
**Gotcha:** `=` is assignment, `==` is comparison. Confusing them is the #1 beginner bug.

### f-strings (the modern way to format text)
```python
name = "Alice"
score = 92.456
print(f"{name} scored {score:.1f}%")   # Alice scored 92.5%
print(f"{name!r}")                      # 'Alice' (repr form)
```
The `:.1f` inside `{}` is a **format spec** — `.1f` means "1 decimal place, fixed-point."

### `input()` always returns a string
```python
age = input("Enter your age: ")   # age is a str, even if user types "25"
age = int(age)                      # you must convert it yourself
```

## 2. Common Mistakes
- Forgetting `input()` returns `str` → doing math on it crashes or gives wrong results (`"25" + "5"` is `"255"`, not `30`).
- Using `int()` on a string containing a decimal point → crashes. Use `float()` first.
- Mixing up `/` (always float) and `//` (floor division).
- Not using f-strings, instead doing `"x is " + str(x)` chains.

## 3. Project: BMI & Unit Converter

**Goal:** A CLI tool that computes BMI and category, and converts °C↔°F and km↔miles.

### Full Solution
```python
def calculate_bmi(weight_kg, height_m):
    """Return BMI given weight in kg and height in meters."""
    if height_m <= 0 or weight_kg <= 0:
        raise ValueError("Height and weight must be positive numbers.")
    return weight_kg / (height_m ** 2)


def bmi_category(bmi):
    """Return the standard WHO BMI category for a given BMI value."""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


def celsius_to_fahrenheit(c):
    return c * 9 / 5 + 32


def fahrenheit_to_celsius(f):
    return (f - 32) * 5 / 9


def km_to_miles(km):
    return km * 0.621371


def miles_to_km(miles):
    return miles / 0.621371


def get_positive_float(prompt):
    """Keep asking until the user gives a valid positive number."""
    while True:
        raw = input(prompt)
        try:
            value = float(raw)
            if value <= 0:
                print("Please enter a positive number.")
                continue
            return value
        except ValueError:
            print("That's not a valid number, try again.")


def main():
    print("=== BMI & Unit Converter ===")
    print("1) Calculate BMI")
    print("2) Convert Celsius <-> Fahrenheit")
    print("3) Convert km <-> miles")
    print("4) Quit")

    while True:
        choice = input("\nChoose an option (1-4): ").strip()

        if choice == "1":
            weight = get_positive_float("Weight (kg): ")
            height = get_positive_float("Height (m): ")
            bmi = calculate_bmi(weight, height)
            category = bmi_category(bmi)
            print(f"Your BMI is {bmi:.1f} ({category})")

        elif choice == "2":
            direction = input("Convert (1) C->F or (2) F->C? ").strip()
            value = float(input("Enter temperature: "))
            if direction == "1":
                print(f"{value}°C = {celsius_to_fahrenheit(value):.1f}°F")
            else:
                print(f"{value}°F = {fahrenheit_to_celsius(value):.1f}°C")

        elif choice == "3":
            direction = input("Convert (1) km->miles or (2) miles->km? ").strip()
            value = get_positive_float("Enter distance: ")
            if direction == "1":
                print(f"{value} km = {km_to_miles(value):.2f} miles")
            else:
                print(f"{value} miles = {miles_to_km(value):.2f} km")

        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid option, please choose 1-4.")


if __name__ == "__main__":
    main()
```

**Why it's structured this way:** logic (BMI math, conversions) is separated from I/O (`input`/`print`). This means you can test `calculate_bmi(70, 1.75)` directly without typing anything — a habit that pays off hugely once you reach Day 21 (testing).

## 4. Practice Exercises
1. Write a function `is_leap_year(year)` returning `True`/`False` (divisible by 4, except centuries unless divisible by 400).
2. Write a tip calculator: given a bill amount and a percentage, print the tip and total.
3. **Challenge:** Extend the BMI tool with a 4th unit conversion of your choice (e.g. kg↔lbs), following the same pattern.

---

# DAY 2 — Strings in Depth

## 1. Concept Explanation

### Strings are immutable sequences
```python
s = "hello"
s[0]          # 'h'
s[-1]         # 'o'  (negative indexing from the end)
s[1:4]        # 'ell' (slicing: start inclusive, end exclusive)
s[::-1]       # 'olleh' (reverse via slice step -1)
s[::2]        # 'hlo' (every 2nd character)
```
`s[0] = "H"` raises `TypeError` — strings can't be modified in place. Any "change" creates a *new* string.

### Useful string methods
```python
"  Hello World  ".strip()          # "Hello World"
"Hello World".lower()               # "hello world"
"Hello World".upper()               # "HELLO WORLD"
"Hello World".replace("World", "Python")  # "Hello Python"
"a,b,c".split(",")                   # ['a', 'b', 'c']
"-".join(["a", "b", "c"])            # "a-b-c"
"Hello".startswith("He")              # True
"Hello".endswith("lo")                 # True
"hello world".title()                   # "Hello World"
"Hello".count("l")                       # 2
"Hello".find("l")                          # 2 (index of first match, -1 if not found)
```

### f-strings and format specs (deeper)
```python
pi = 3.14159265
f"{pi:.2f}"      # '3.14'          -- 2 decimal places
f"{pi:10.2f}"    # '      3.14'    -- padded to width 10
f"{42:05d}"       # '00042'         -- zero-padded integer
f"{'hi':>10}"      # '        hi'    -- right-align in width 10
f"{'hi':<10}|"      # 'hi        |'   -- left-align
f"{'hi':^10}|"       # '    hi    |'   -- center-align
f"{1000000:,}"        # '1,000,000'     -- thousands separator
```

### Iterating over strings
```python
for char in "abc":
    print(char)   # a, b, c on separate lines

"a" in "abc"       # True — substring/character membership test
```

## 2. Common Mistakes
- Trying to mutate a string directly (`s[0] = "x"`) — always build a *new* string instead.
- `.split()` with no arguments splits on **any whitespace** and collapses multiples, while `.split(",")` splits on exactly `","` and keeps empty strings between consecutive commas.
- Off-by-one errors in slicing — end index is *exclusive*.
- Comparing strings case-sensitively without normalizing (`"Hi" != "hi"`).

## 3. Project: Text Analyzer

**Goal:** Given a block of text, report word count, character count (no spaces), the most common word, the longest word, and whether it's a palindrome (ignoring case/punctuation/spaces).

### Full Solution
```python
import string


def clean_words(text):
    """Split text into lowercase words, stripped of surrounding punctuation."""
    words = text.split()
    cleaned = []
    for w in words:
        w = w.strip(string.punctuation).lower()
        if w:  # skip empty strings left over from pure-punctuation "words"
            cleaned.append(w)
    return cleaned


def word_count(text):
    return len(clean_words(text))


def char_count_no_spaces(text):
    return len(text.replace(" ", ""))


def most_common_word(text):
    words = clean_words(text)
    if not words:
        return None
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    # max() with a key function finds the word with the highest count
    return max(counts, key=counts.get)


def longest_word(text):
    words = clean_words(text)
    if not words:
        return None
    return max(words, key=len)


def is_palindrome(text):
    """Check if text reads the same forwards and backwards,
    ignoring case, spaces, and punctuation."""
    normalized = "".join(ch.lower() for ch in text if ch.isalnum())
    return normalized == normalized[::-1]


def analyze(text):
    return {
        "word_count": word_count(text),
        "char_count_no_spaces": char_count_no_spaces(text),
        "most_common_word": most_common_word(text),
        "longest_word": longest_word(text),
        "is_palindrome": is_palindrome(text),
        "reversed_text": text[::-1],
    }


def print_report(text):
    result = analyze(text)
    print("\n--- Text Analysis Report ---")
    print(f"Word count:            {result['word_count']}")
    print(f"Characters (no spaces): {result['char_count_no_spaces']}")
    print(f"Most common word:       {result['most_common_word']}")
    print(f"Longest word:           {result['longest_word']}")
    print(f"Is palindrome:          {result['is_palindrome']}")
    print(f"Reversed text:          {result['reversed_text']}")


def main():
    print("=== Text Analyzer ===")
    while True:
        text = input("\nEnter text to analyze (or 'quit' to exit): ")
        if text.lower() == "quit":
            break
        if not text.strip():
            print("Please enter some text.")
            continue
        print_report(text)


if __name__ == "__main__":
    main()
```

**Design notes:**
- `clean_words()` centralizes tokenization so every other function reuses it — avoids 5 slightly-different tokenization bugs.
- `max(counts, key=counts.get)` is an important idiom: `key=` tells `max()` *what to compare by*, not what to return.
- `is_palindrome` strips punctuation with `ch.isalnum()` rather than `string.punctuation`, because it also needs to drop spaces — two valid ways to filter characters depending on what you need.

## 4. Practice Exercises
1. Write `count_vowels(text)` returning the number of vowels (a, e, i, o, u, case-insensitive).
2. Write a function that capitalizes the first letter of every word *except* small words like "a", "the", "of" (simple book-title casing).
3. **Challenge:** Add a feature reporting the **top 3** most common words, not just one (hint: sort the `counts` dict by value — a preview of Day 20's `sorted(key=...)`).

---

# DAY 3 — Lists & Tuples

## 1. Concept Explanation

### Lists: ordered, mutable
```python
fruits = ["apple", "banana", "cherry"]
fruits[0]            # 'apple'
fruits[-1]            # 'cherry'
fruits.append("date")  # adds to the end
fruits.insert(1, "x")   # insert at index 1
fruits.remove("x")       # removes first matching value
fruits.pop()               # removes & returns last item
fruits.pop(0)                # removes & returns item at index 0
len(fruits)                    # number of items
```

### Slicing (same rules as strings)
```python
nums = [10, 20, 30, 40, 50]
nums[1:3]     # [20, 30]
nums[:2]       # [10, 20]
nums[2:]        # [30, 40, 50]
nums[::-1]        # [50, 40, 30, 20, 10]
```

### Sorting
```python
nums = [3, 1, 4, 1, 5]
nums.sort()               # in-place, mutates nums -> [1, 1, 3, 4, 5]
nums.sort(reverse=True)    # [5, 4, 3, 1, 1]

sorted_copy = sorted(nums)   # returns a NEW sorted list, original unchanged
```
**Rule of thumb:** `.sort()` = mutates, returns `None`. `sorted()` = returns a new list, leaves original untouched. Prefer `sorted()` unless you specifically want to mutate.

### The mutability trap
```python
a = [1, 2, 3]
b = a           # b is the SAME list object as a, not a copy!
b.append(4)
print(a)        # [1, 2, 3, 4]  <- a changed too!

c = a.copy()    # or a[:], or list(a) — an actual independent copy
c.append(5)
print(a)         # unaffected
```

### Tuples: ordered, immutable
```python
point = (3, 4)
point[0]          # 3
point[0] = 10       # TypeError! tuples can't be modified

x, y = point          # unpacking
print(x, y)             # 3 4

def min_max(numbers):
    return min(numbers), max(numbers)

lo, hi = min_max([4, 1, 9, 2])   # lo=1, hi=9
```

**Why tuples exist when lists can do more:** immutability makes tuples safe as dict keys, safe to pass around without fear something mutates them, and signals intent ("this is a fixed record", like a coordinate or an RGB triple).

### List of tuples — a very common pattern
```python
tasks = [("Buy milk", False), ("Clean house", True)]
for name, done in tasks:
    status = "✓" if done else " "
    print(f"[{status}] {name}")
```

## 2. Common Mistakes
- Assuming `b = a` copies a list — it doesn't; both names point to the same object.
- Using `.sort()` and expecting a return value (`x = nums.sort()` makes `x = None`).
- Trying to mutate a tuple or `.append()` to one (tuples have no `.append`).
- Modifying a list while iterating over it (`for x in lst: lst.remove(x)`) — causes skipped elements. Iterate over a copy (`for x in lst[:]`) if you need to remove during iteration.

## 3. Project: In-Memory To-Do List (CLI)

**Goal:** Add, remove, list, and mark-done tasks, stored as a list of `(task, done)` tuples, all within a single running session (no file saving yet — that's Day 9).

### Full Solution
```python
def display_tasks(tasks):
    if not tasks:
        print("No tasks yet. Add one!")
        return
    print("\n--- Your Tasks ---")
    for i, (task, done) in enumerate(tasks, start=1):
        status = "✓" if done else " "
        print(f"{i}. [{status}] {task}")


def add_task(tasks, description):
    tasks.append((description, False))
    return tasks


def remove_task(tasks, index):
    """index is 1-based, as shown to the user."""
    if 1 <= index <= len(tasks):
        removed = tasks.pop(index - 1)
        print(f"Removed: {removed[0]}")
    else:
        print("Invalid task number.")
    return tasks


def mark_done(tasks, index):
    if 1 <= index <= len(tasks):
        task_name, _ = tasks[index - 1]
        # tuples can't be mutated, so we REPLACE the tuple at that index
        tasks[index - 1] = (task_name, True)
    else:
        print("Invalid task number.")
    return tasks


def get_index(prompt, tasks):
    try:
        index = int(input(prompt))
        return index
    except ValueError:
        print("Please enter a valid number.")
        return None


def main():
    tasks = []  # list of (description: str, done: bool) tuples

    menu = """
=== To-Do List ===
1. View tasks
2. Add task
3. Remove task
4. Mark task as done
5. Quit
"""
    while True:
        print(menu)
        choice = input("Choose an option (1-5): ").strip()

        if choice == "1":
            display_tasks(tasks)

        elif choice == "2":
            description = input("Task description: ").strip()
            if description:
                add_task(tasks, description)
                print("Task added.")
            else:
                print("Task description can't be empty.")

        elif choice == "3":
            display_tasks(tasks)
            if tasks:
                index = get_index("Task number to remove: ", tasks)
                if index is not None:
                    remove_task(tasks, index)

        elif choice == "4":
            display_tasks(tasks)
            if tasks:
                index = get_index("Task number to mark done: ", tasks)
                if index is not None:
                    mark_done(tasks, index)
                    print("Task marked as done.")

        elif choice == "5":
            print("Goodbye!")
            break

        else:
            print("Invalid option, choose 1-5.")


if __name__ == "__main__":
    main()
```

**Design notes:**
- `enumerate(tasks, start=1)` gives both index and item, starting the count at 1 for user-friendly display, instead of manual `range(len(tasks))` indexing.
- Since a tuple `(name, done)` can't be mutated in place, `mark_done` **replaces the whole tuple** at that index — a concrete illustration of Day 2/3 immutability, and it foreshadows why Day 11 (OOP) will feel much nicer for "update one field" problems.
- Functions take `tasks` as a parameter and return it — even though lists mutate in place, this keeps signatures explicit about what they operate on (a good habit before classes make this implicit via `self`).

## 4. Practice Exercises
1. Add a `priority` field: change tuples to `(task, done, priority)` and add a menu option to list tasks sorted by priority.
2. Write `completed_count(tasks)` returning how many tasks are done, using nothing but a loop.
3. **Challenge:** Add an "edit task description" option (given a task number, let the user retype the description) — reuse the tuple-replacement pattern from `mark_done`.

---

## Quick Self-Check Before Moving to Day 4

You should be able to, without looking anything up:
- Explain the difference between `/` and `//`, and between `.sort()` and `sorted()`.
- Slice a string or list to get the last 3 elements, and to reverse it.
- Explain why `b = a` doesn't copy a list, and how to actually copy one.
- Explain why tuples can't have items reassigned, but a list *of* tuples can have its tuples *replaced*.

If any of those feel shaky, redo the relevant practice exercise before Day 4 (dictionaries & sets) — dictionaries build directly on this mutability/immutability foundation (only immutable types can be dict keys).

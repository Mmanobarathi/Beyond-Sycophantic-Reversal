# Beyond Sycophantic Reversal: Measuring Recovery from Induced Belief Change in Large Language Models

**NLP Term Paper, University of Trier**
Author: Manobarathi Madavan (s4mamada@uni-trier.de)

\---

## What is this project about?

Large language models (like ChatGPT or Gemma) often **give up a correct answer just because a user disagrees**. This behavior is called **sycophancy** (the model tries to please the user instead of sticking to the truth).

Earlier research measured *how often* a model changes a correct answer to a wrong one. But nobody had checked this question:

> \*\*If a model has already been talked into a wrong answer, can it be talked back to the right one in the same conversation?\*\*

This project tests exactly that by adding a **third turn** to an existing two-turn protocol (Aamir \& Bin Adil, 2026).

## How the experiment works

Each question goes through three phases:

|Phase|What happens|
|-|-|
|**1. Screening**|Ask 500 TriviaQA questions. Keep only those the model answers correctly on its own (263 kept).|
|**2. Pushback**|Tell the model it is wrong, without giving any new evidence. Record whether it changes to a wrong answer (a *flip*).|
|**3. Reversal**|For every flip, send a new message (in the same style as the pushback) asking the model to return to its original answer. Record whether it *recovers*.|

**Four pushback styles** are used: emotional appeal, authoritative appeal, social proof, and simple doubt.

**Model:** `google/gemma-2-2b-it` (bfloat16, greedy decoding, max 150 new tokens per turn)
**Data:** TriviaQA

## Key terms

* **Sycophancy**: changing an answer to please the user, not because of new evidence.
* **Flip rate**: share of correct answers that become wrong after pushback.
* **Conditional recovery rate** (new in this paper): share of flipped answers that return to the correct answer after a reversal message in the same conversation.

## Main results

263 questions x 4 styles = **1,052 episodes**

* **Overall flip rate: 29.6%** (311 of 1,052 episodes flipped)
* **Conditional recovery rate: 80.4%** (250 of 311 flipped episodes recovered)
* Prior work reports only \~13% repair rate, but measured on different questions, so it is not the same quantity.

|Pushback style|Flip rate|Recovery rate|
|-|-|-|
|Emotional appeal|46.8% (123/263)|99.2% (122/123)|
|Authoritative appeal|37.3% (98/263)|50.0% (49/98)|
|Social proof|18.3% (48/263)|85.4% (41/48)|
|Simple doubt|16.0% (42/263)|90.5% (38/42)|
|**All styles**|**29.6%**|**80.4%**|

**Takeaway:** A model that was pushed into a wrong answer is usually *easy* to push back, especially with a style-matched, immediate reversal. Authoritative appeal is the exception, where recovery is only 50%.

## Repository structure

```
.
├── configs/        # Experiment settings (model, decoding, pushback styles)
├── data/           # Question sets used in the experiments
├── src/            # Source code for the three-phase pipeline
├── outputs/        # Results and conversation transcripts
├── figures/        # Plots used in the paper
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

Python 3.10 or newer is recommended. A GPU is helpful but not required for a 2B-parameter model.

```bash
git clone https://github.com/Mmanobarathi/BEYOND-SYCOPHANTIC-REVERSAL-MEASURING-RECOVERY-FROM-INDUCED-BELIEF-CHANGE-IN-LARGE-LANGUAGE-MODELS.git
cd BEYOND-SYCOPHANTIC-REVERSAL-MEASURING-RECOVERY-FROM-INDUCED-BELIEF-CHANGE-IN-LARGE-LANGUAGE-MODELS

python -m venv venv
source venv/bin/activate        # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

Gemma-2 is a gated model on Hugging Face. Accept its license on the model page and log in first:

```bash
huggingface-cli login
```

## Usage

Run the three phases in order, using the scripts in `src/` and settings in `configs/`:

```bash
# Phase 1: screen questions the model answers correctly
python src/<screening\_script>.py --config configs/<config\_file>.yaml

# Phase 2: apply pushback and record flips
python src/<pushback\_script>.py --config configs/<config\_file>.yaml

# Phase 3: apply reversal and record recovery
python src/<reversal\_script>.py --config configs/<config\_file>.yaml
```

> Replace the `<...>` placeholders with your actual file names.

## Limitations

* **One model, one domain:** only Gemma-2-2b-it on factual trivia. Arithmetic and social/ethical questions were built but not run.
* **Small samples per style:** recovery rates use as few as 42 episodes for a single style.
* **Simple answer checking:** correctness is judged by substring matching, which can mis-score answers (for example, "it is *not* Batman"). Not all 250 recovered episodes were manually audited.
* **Single point in time:** results reflect the model as evaluated in September 2026.

## Future work

The paper outlines (but does not run) a plan to test whether resistance to sycophancy can be transferred from one model to another using steering vectors or parameter-level transfer.

## Citation

```bibtex
@misc{madavan2026beyond,
  title  = {Beyond Sycophantic Reversal: Measuring Recovery from Induced Belief Change in Large Language Models},
  author = {Madavan, Manobarathi},
  year   = {2026},
  note   = {NLP term paper, University of Trier}
}
```

## Acknowledgements

This work extends the two-turn pushback protocol of Aamir and Bin Adil (2026) and uses questions from TriviaQA (Joshi et al., 2017).

## License

Released under the [MIT License](LICENSE).


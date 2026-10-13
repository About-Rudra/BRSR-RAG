# Evaluation set

`questions.jsonl` has one JSON object per line. The three rows in it are **placeholders**
(questions starting with `<` are skipped by the runner); replace them with your own questions,
written by reading the indexed PDFs. Do not generate questions or expected answers with an LLM.

| Field | Meaning |
|---|---|
| `id` | Unique id, e.g. `Q01` |
| `question` | The question exactly as a user would ask it |
| `regimes` | List of regime keys it needs: `sebi_brsr`, `cpcb_epr`, `ccts` (empty for `not_in_corpus`) |
| `expected_answer` | The correct answer in your own words (used when you mark correctness) |
| `expected_source` | File name of the PDF that contains the answer (as listed in the app's Sources tab) |
| `expected_page` | Page number in that PDF where the answer is (0 if not applicable) |
| `type` | `single` (one regime), `multi_regime` (needs two or more), or `not_in_corpus` (answer is not in the documents) |

## Running

```
python -m eval.run_eval                  # baseline and agent, same LLM provider/model
python -m eval.run_eval --llm-judge      # optional automatic, secondary correctness column
```

Results go to `eval/results/<timestamp>/`: `outputs.jsonl` (full answers, plans, verdicts, traces)
and `summary.csv` (automatic metrics). Mark `correct_baseline` and `correct_agent` by hand
(`y`, `n` or `partial`) by comparing each answer with your expected answer, then:

```
python -m eval.summarise eval/results/<timestamp>/summary.csv
```

Automatic metrics: has a citation; cited source matches `expected_source`; cited page matches
`expected_page` (agent only); claims rejected by the verifier; tool calls, LLM calls and latency;
and for `not_in_corpus` questions, whether the system said the answer was not found.

On a free Gemini key, each agent question takes ~8-20 LLM calls, so 30 questions take a while and
may hit the daily quota; run in batches with `--limit` if needed.

#!/usr/bin/env python3
"""
Generate a CSV of (string_prefix, chars_since_nl) pairs for newline-distance analysis.

Each row's `string` is a prefix of a wrapped text such that tokenizing it with the
same tokenizer yields the labelled token as the final token.

All fields are enclosed in double quotes so that newlines and commas within
strings are handled correctly by any RFC 4180-compliant CSV parser.
"""

import csv
import textwrap
from itertools import islice

import pandas as pd
import typer
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def wrap_preserve_newlines(text: str, width: int) -> str:
    wrapper = textwrap.TextWrapper(width=width)
    out: list[str] = []
    for line in text.splitlines(keepends=False):
        if line.strip() == "":
            out.append(line)
        else:
            out.extend(wrapper.wrap(line))
    return "\n".join(out)


def labeled_prefixes(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    max_seq_len: int = 2048,
) -> list[dict]:
    enc = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_len,
        padding=False,
        return_offsets_mapping=True,
    )

    special_ids = set(getattr(tokenizer, "all_special_ids", ()))
    last_nl_pos = -1
    rows: list[dict] = []

    for tid, (s, e) in zip(enc["input_ids"], enc["offset_mapping"]):
        tid = int(tid)
        if tid in special_ids or s == e:
            continue

        nl = text.rfind("\n", s, e)
        if nl != -1:
            last_nl_pos = nl

        rows.append(
            {
                "string": text[:e],
                "chars_since_nl": e - last_nl_pos - 1,
            }
        )

    return rows


def generate_labeled_data(
    tokenizer: str | PreTrainedTokenizerBase = "google/gemma-2-9b",
    dataset_name: str = "monology/pile-uncopyrighted",
    output_csv: str = "newline_labeled.csv",
    line_length: int = 80,
    num_samples: int = 100,
    min_lines: int = 5,
    max_seq_len: int = 2048,
    text_key: str = "text",
) -> pd.DataFrame:
    if isinstance(tokenizer, str):
        typer.echo(f"Loading tokenizer: {tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)

    typer.echo(
        f"Streaming {dataset_name} (filter: len > {line_length * min_lines}, no newlines)"
    )
    stream = load_dataset(dataset_name, split="train", streaming=True)
    stream = stream.filter(
        lambda x: len(x[text_key]) > line_length * min_lines and "\n" not in x[text_key]
    )
    dataset = Dataset.from_generator(lambda: islice(stream, num_samples))
    typer.echo(f"Materialised {len(dataset)} samples")

    if len(dataset) == 0:
        typer.echo("No samples survived the filter.")
        return pd.DataFrame(columns=["string", "chars_since_nl"])

    all_rows: list[dict] = []
    for idx in tqdm(range(len(dataset)), desc="Labelling"):
        raw = dataset[idx][text_key]
        wrapped = wrap_preserve_newlines(raw, width=line_length)
        all_rows.extend(labeled_prefixes(tokenizer, wrapped, max_seq_len))

    df = pd.DataFrame(all_rows, columns=["string", "chars_since_nl"])
    df.to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL)

    s = df["chars_since_nl"]
    typer.echo(f"\nSaved {len(df):,} rows → {output_csv}")
    typer.echo(
        f"  chars_since_nl:  mean={s.mean():.1f}  std={s.std():.1f}  min={s.min()}  max={s.max()}"
    )
    return df


app = typer.Typer()


@app.command()
def main(
    tokenizer: str = typer.Option(
        "google/gemma-2-9b", help="HuggingFace tokenizer name or path."
    ),
    dataset: str = typer.Option(
        "monology/pile-uncopyrighted", help="HuggingFace dataset identifier."
    ),
    output_csv: str = typer.Option("newline_labeled.csv", help="Output CSV path."),
    line_length: int = typer.Option(
        80, help="Wrap text to this many characters per line."
    ),
    num_samples: int = typer.Option(100, help="Number of post-filter samples to use."),
    min_lines: int = typer.Option(
        5, help="Reject texts shorter than line_length × min_lines."
    ),
    max_seq_len: int = typer.Option(2048, help="Tokenizer truncation limit."),
    text_key: str = typer.Option(
        "text", help="Column name for raw text in the dataset."
    ),
) -> None:
    """Generate per-token newline-distance labels as a CSV."""
    generate_labeled_data(
        tokenizer=tokenizer,
        dataset_name=dataset,
        output_csv=output_csv,
        line_length=line_length,
        num_samples=num_samples,
        min_lines=min_lines,
        max_seq_len=max_seq_len,
        text_key=text_key,
    )


if __name__ == "__main__":
    app()

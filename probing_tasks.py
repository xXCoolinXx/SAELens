"""
Manifold Probing Script for SMIXAE

Generates template-based probing datasets with systematically varied features
(months, days, hours, names, numbers, body parts, etc.), runs them through
the model + SAE, and plots bottleneck coordinates colored by feature values
to discover geometric structures (rings, lines, sheets, clusters).

Uses multivariate Fisher score (trace(S_w^{-1} S_b)) to find experts that
best separate feature classes in 3D bottleneck space.

Inspired by "Shape Happens" (Tiblias et al., 2025) probing methodology.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from transformer_lens import HookedTransformer

from sae_lens import SAE

# ============================================================
# Config
# ============================================================
MODEL_NAME = "gemma-2-2b"
HOOK_NAME = "blocks.12.hook_resid_post"
CHECKPOINT_PATH = (
    "/scratch/Collin/SAELens/checkpoints/apuvg3w9/final_250003456"  # UPDATE THIS
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("manifold_probing_results")
OUTPUT_DIR.mkdir(exist_ok=True)

N_EXPERTS_TO_PLOT = 25


# ============================================================
# 1. Define probing tasks
# ============================================================
@dataclass
class ProbingTask:
    name: str
    description: str
    templates: list[str]
    feature_name: str
    feature_values: list[str]
    feature_numeric: list[float] = field(default_factory=list)
    cyclic: bool = False
    target_position: str = "last"


def build_probing_tasks() -> list[ProbingTask]:
    tasks = []

    # --- Months of the year (expected: ring) ---
    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    tasks.append(
        ProbingTask(
            name="months",
            description="Months of the year - expected ring/circle",
            templates=[
                "The month is {feature}.",
                "It happened in {feature} last year.",
                "We plan to visit in {feature}.",
                "The report is due in {feature}.",
                "She was born in {feature}.",
                "The conference takes place in {feature}.",
                "Sales peaked in {feature}.",
                "The weather in {feature} is usually mild.",
            ],
            feature_name="month",
            feature_values=months,
            feature_numeric=[float(i) for i in range(12)],
            cyclic=True,
        )
    )

    # --- Days of the week (expected: ring) ---
    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    tasks.append(
        ProbingTask(
            name="days_of_week",
            description="Days of the week - expected ring/circle",
            templates=[
                "Today is {feature}.",
                "The meeting is on {feature}.",
                "We always go shopping on {feature}.",
                "The deadline is {feature}.",
                "I have a doctor's appointment on {feature}.",
                "The store is closed on {feature}.",
                "{feature} is my favorite day.",
                "The game is scheduled for {feature}.",
            ],
            feature_name="day",
            feature_values=days,
            feature_numeric=[float(i) for i in range(7)],
            cyclic=True,
        )
    )

    # --- Hours of the day (expected: ring) ---
    hours_str = [
        "1 AM",
        "2 AM",
        "3 AM",
        "4 AM",
        "5 AM",
        "6 AM",
        "7 AM",
        "8 AM",
        "9 AM",
        "10 AM",
        "11 AM",
        "12 PM",
        "1 PM",
        "2 PM",
        "3 PM",
        "4 PM",
        "5 PM",
        "6 PM",
        "7 PM",
        "8 PM",
        "9 PM",
        "10 PM",
        "11 PM",
        "12 AM",
    ]
    tasks.append(
        ProbingTask(
            name="hours",
            description="Hours of the day - expected ring/circle",
            templates=[
                "The time is {feature}.",
                "We arrived at {feature}.",
                "The train departs at {feature}.",
                "The alarm is set for {feature}.",
                "The store opens at {feature}.",
                "I woke up at {feature}.",
            ],
            feature_name="hour",
            feature_values=hours_str,
            feature_numeric=[float(i) for i in range(24)],
            cyclic=True,
        )
    )

    # --- Time durations / units (expected: line/plume) ---
    time_units = [
        "nanosecond",
        "microsecond",
        "millisecond",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "decade",
        "century",
        "millennium",
    ]
    tasks.append(
        ProbingTask(
            name="time_units",
            description="Time duration units - expected line/plume (log scale)",
            templates=[
                "It lasted about one {feature}.",
                "The process takes one {feature}.",
                "Wait for one {feature}.",
                "The event lasted a {feature}.",
                "One {feature} passed quickly.",
                "It takes roughly a {feature}.",
            ],
            feature_name="unit",
            feature_values=time_units,
            feature_numeric=[float(i) for i in range(len(time_units))],
            cyclic=False,
        )
    )

    # --- Numbers 1-100 (expected: line or curve) ---
    numbers = [str(i) for i in range(1, 101)]
    tasks.append(
        ProbingTask(
            name="numbers",
            description="Numbers 1-100 - expected line or curve",
            templates=[
                "There are {feature} items in the box.",
                "She counted {feature} stars.",
                "The answer is {feature}.",
                "He scored {feature} points.",
                "We need {feature} more.",
            ],
            feature_name="number",
            feature_values=numbers,
            feature_numeric=[float(i) for i in range(1, 101)],
            cyclic=False,
        )
    )

    # --- Common first names (expected: clusters by gender/origin) ---
    names = [
        "James",
        "Mary",
        "John",
        "Patricia",
        "Robert",
        "Jennifer",
        "Michael",
        "Linda",
        "William",
        "Elizabeth",
        "David",
        "Barbara",
        "Richard",
        "Susan",
        "Joseph",
        "Jessica",
        "Thomas",
        "Sarah",
        "Daniel",
        "Karen",
        "Matthew",
        "Nancy",
        "Anthony",
        "Lisa",
        "Mohammed",
        "Fatima",
        "Wei",
        "Mei",
        "Hiroshi",
        "Yuki",
        "Carlos",
        "Maria",
        "Pierre",
        "Sophie",
        "Ivan",
        "Olga",
    ]
    tasks.append(
        ProbingTask(
            name="first_names",
            description="Common first names - expected clusters",
            templates=[
                "My name is {feature}.",
                "{feature} went to the store.",
                "I talked to {feature} yesterday.",
                "Have you met {feature}?",
                "{feature} is a good friend.",
                "The letter was addressed to {feature}.",
            ],
            feature_name="name",
            feature_values=names,
            feature_numeric=[float(i) for i in range(len(names))],
            cyclic=False,
        )
    )

    # --- Body parts (expected: sheet/spatial structure) ---
    body_parts = [
        "head",
        "forehead",
        "eye",
        "nose",
        "mouth",
        "chin",
        "ear",
        "neck",
        "shoulder",
        "arm",
        "elbow",
        "wrist",
        "hand",
        "finger",
        "chest",
        "stomach",
        "back",
        "hip",
        "thigh",
        "knee",
        "shin",
        "ankle",
        "foot",
        "toe",
    ]
    tasks.append(
        ProbingTask(
            name="body_parts",
            description="Body parts - expected sheet/spatial manifold",
            templates=[
                "The pain was in my {feature}.",
                "She pointed to her {feature}.",
                "He injured his {feature}.",
                "The doctor examined my {feature}.",
                "I felt a tingling in my {feature}.",
                "The tattoo is on my {feature}.",
            ],
            feature_name="part",
            feature_values=body_parts,
            feature_numeric=[float(i) for i in range(len(body_parts))],
            cyclic=False,
        )
    )

    # --- Colors (expected: ring or curve through color space) ---
    colors = [
        "red",
        "orange",
        "yellow",
        "green",
        "cyan",
        "blue",
        "purple",
        "pink",
        "brown",
        "black",
        "white",
        "gray",
    ]
    tasks.append(
        ProbingTask(
            name="colors",
            description="Colors - expected ring through color space",
            templates=[
                "The wall is painted {feature}.",
                "She wore a {feature} dress.",
                "The car is {feature}.",
                "His favorite color is {feature}.",
                "The {feature} light was blinding.",
                "They chose {feature} for the logo.",
            ],
            feature_name="color",
            feature_values=colors,
            feature_numeric=[float(i) for i in range(len(colors))],
            cyclic=True,
        )
    )

    # --- Compass directions (expected: ring) ---
    directions = [
        "north",
        "northeast",
        "east",
        "southeast",
        "south",
        "southwest",
        "west",
        "northwest",
    ]
    tasks.append(
        ProbingTask(
            name="directions",
            description="Compass directions - expected ring",
            templates=[
                "The wind blows from the {feature}.",
                "We headed {feature}.",
                "The city is to the {feature}.",
                "The storm is coming from the {feature}.",
                "Turn {feature} at the intersection.",
            ],
            feature_name="direction",
            feature_values=directions,
            feature_numeric=[float(i) for i in range(len(directions))],
            cyclic=True,
        )
    )

    # --- Ordinal sizes (expected: line) ---
    sizes = [
        "microscopic",
        "tiny",
        "small",
        "medium",
        "large",
        "huge",
        "enormous",
        "gigantic",
        "colossal",
    ]
    tasks.append(
        ProbingTask(
            name="sizes",
            description="Ordinal sizes - expected line",
            templates=[
                "The object was {feature}.",
                "It was a {feature} building.",
                "The {feature} animal approached.",
                "They found a {feature} rock.",
                "The portion was {feature}.",
            ],
            feature_name="size",
            feature_values=sizes,
            feature_numeric=[float(i) for i in range(len(sizes))],
            cyclic=False,
        )
    )

    # --- Temperatures (expected: line) ---
    temperatures = [
        "freezing",
        "cold",
        "cool",
        "mild",
        "warm",
        "hot",
        "scorching",
        "boiling",
    ]
    tasks.append(
        ProbingTask(
            name="temperatures",
            description="Temperature words - expected line",
            templates=[
                "The water was {feature}.",
                "It felt {feature} outside.",
                "The soup was {feature}.",
                "The room was {feature}.",
                "The weather today is {feature}.",
            ],
            feature_name="temp",
            feature_values=temperatures,
            feature_numeric=[float(i) for i in range(len(temperatures))],
            cyclic=False,
        )
    )

    # --- Countries (expected: spatial/cluster structure) ---
    countries = [
        "France",
        "Germany",
        "Italy",
        "Spain",
        "England",
        "Japan",
        "China",
        "India",
        "Korea",
        "Thailand",
        "Brazil",
        "Mexico",
        "Argentina",
        "Canada",
        "Australia",
        "Egypt",
        "Nigeria",
        "Kenya",
        "Russia",
        "Turkey",
    ]
    tasks.append(
        ProbingTask(
            name="countries",
            description="Countries - expected spatial/geographic structure",
            templates=[
                "She traveled to {feature}.",
                "The food from {feature} is delicious.",
                "He was born in {feature}.",
                "The capital of {feature} is beautiful.",
                "{feature} is known for its culture.",
            ],
            feature_name="country",
            feature_values=countries,
            feature_numeric=[float(i) for i in range(len(countries))],
            cyclic=False,
        )
    )

    return tasks


# ============================================================
# 2. Generate texts and track feature positions
# ============================================================
@dataclass
class ProbingExample:
    text: str
    feature_value: str
    feature_numeric: float
    template_idx: int
    target_token_idx: int = -1


def generate_examples(task: ProbingTask, tokenizer) -> list[ProbingExample]:
    examples = []

    for template_idx, template in enumerate(task.templates):
        for val_idx, value in enumerate(task.feature_values):
            text = template.format(feature=value)

            prefix = template.split("{feature}")[0] + value
            prefix_tokens = tokenizer.encode(prefix)
            full_tokens = tokenizer.encode(text)

            target_idx = len(prefix_tokens) - 1
            target_idx = min(target_idx, len(full_tokens) - 1)

            numeric = (
                task.feature_numeric[val_idx]
                if task.feature_numeric
                else float(val_idx)
            )

            examples.append(
                ProbingExample(
                    text=text,
                    feature_value=value,
                    feature_numeric=numeric,
                    template_idx=template_idx,
                    target_token_idx=target_idx,
                )
            )

    return examples


# ============================================================
# 3. Collect activations
# ============================================================
def collect_activations(
    model: HookedTransformer,
    examples: list[ProbingExample],
    hook_name: str,
    batch_size: int = 64,
) -> torch.Tensor:
    all_acts = []

    for batch_start in range(0, len(examples), batch_size):
        batch = examples[batch_start : batch_start + batch_size]
        texts = [ex.text for ex in batch]

        tokens = model.to_tokens(texts, prepend_bos=True)
        seq_len = tokens.shape[1]

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)

        acts = cache[hook_name]  # (batch, seq, d_model)

        for i, ex in enumerate(batch):
            target_idx = min(ex.target_token_idx + 1, seq_len - 1)
            all_acts.append(acts[i, target_idx, :].cpu())

    return torch.stack(all_acts, dim=0)


# ============================================================
# 4. Run through SAE and extract bottleneck coordinates
# ============================================================
def get_bottleneck_coords(
    sae: SAE, activations: torch.Tensor, batch_size: int = 256
) -> np.ndarray:
    all_z = []

    for batch_start in range(0, len(activations), batch_size):
        batch = activations[batch_start : batch_start + batch_size].to(sae.device)
        with torch.no_grad():
            z = sae.encode(batch)
        all_z.append(z.cpu().numpy())

    return np.concatenate(all_z, axis=0)


# ============================================================
# 5. Multivariate Fisher score for expert finding
# ============================================================
def fisher_score_multivariate(
    z_expert: np.ndarray, class_labels: np.ndarray, n_classes: int
) -> float:
    """
    Multivariate Fisher criterion: trace(S_w^{-1} S_b).
    Measures how well the 3D bottleneck separates classes.
    """
    d = z_expert.shape[1]
    overall_mean = z_expert.mean(axis=0)

    S_w = np.zeros((d, d))
    S_b = np.zeros((d, d))

    classes_present = 0

    for c in range(n_classes):
        mask = class_labels == c
        if mask.sum() < 2:
            continue

        classes_present += 1
        z_c = z_expert[mask]
        mean_c = z_c.mean(axis=0)
        n_c = len(z_c)

        # Within-class scatter
        diff = z_c - mean_c
        S_w += diff.T @ diff

        # Between-class scatter
        d_vec = (mean_c - overall_mean).reshape(-1, 1)
        S_b += n_c * (d_vec @ d_vec.T)

    if classes_present < 3:
        return 0.0

    # Regularize S_w
    S_w += np.eye(d) * 1e-6

    return float(np.trace(np.linalg.solve(S_w, S_b)))


def find_responsive_experts(
    z_np: np.ndarray,
    examples: list[ProbingExample],
    task: ProbingTask,
    n_experts: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find experts whose 3D bottleneck best separates feature classes
    using multivariate Fisher score.

    Returns (top_expert_ids, all_scores).
    """
    value_to_class = {v: i for i, v in enumerate(task.feature_values)}
    class_labels = np.array([value_to_class[ex.feature_value] for ex in examples])
    n_classes = len(task.feature_values)

    n_exp = z_np.shape[1]
    scores = np.zeros(n_exp)

    for expert_id in range(n_exp):
        norms = np.linalg.norm(z_np[:, expert_id, :], axis=-1)
        active = norms > 1e-3

        if active.sum() < 10:
            continue

        z_e = z_np[active, expert_id, :]
        labels_e = class_labels[active]

        if len(np.unique(labels_e)) < 3:
            continue

        scores[expert_id] = fisher_score_multivariate(z_e, labels_e, n_classes)

    top_idx = np.argsort(scores)[::-1][:n_experts]
    return top_idx, scores


# ============================================================
# 6. Plotting
# ============================================================
def get_colorscale(task: ProbingTask):
    if task.cyclic:
        return "HSV"
    return "Viridis"


def plot_task_experts(
    task: ProbingTask,
    expert_ids: np.ndarray,
    fisher_scores: np.ndarray,
    z_np: np.ndarray,
    examples: list[ProbingExample],
    activation_freq: np.ndarray,
    mean_norm_per_expert: np.ndarray,
    output_dir: Path,
):
    n = len(expert_ids)
    cols = min(5, n)
    rows = (n + cols - 1) // cols

    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "scene"}] * cols for _ in range(rows)],
        subplot_titles=[
            f"Expert {eid} (Fisher: {fisher_scores[eid]:.1f}, "
            f"freq: {activation_freq[eid]:.2%})"
            for eid in expert_ids
        ],
        horizontal_spacing=0.02,
        vertical_spacing=0.08,
    )

    feature_values = np.array([ex.feature_value for ex in examples])
    feature_numeric = np.array([ex.feature_numeric for ex in examples])

    colorscale = get_colorscale(task)

    for idx, expert_id in enumerate(expert_ids):
        row = idx // cols + 1
        col = idx % cols + 1

        norms = np.linalg.norm(z_np[:, expert_id, :], axis=-1)
        active = norms > 1e-3
        n_active = active.sum()

        if n_active < 5:
            continue

        z_e = z_np[active, expert_id, :]
        labels_e = feature_values[active]
        numeric_e = feature_numeric[active]

        max_points = 5000
        if n_active > max_points:
            subsample = np.random.choice(n_active, max_points, replace=False)
            z_e = z_e[subsample]
            labels_e = labels_e[subsample]
            numeric_e = numeric_e[subsample]

        hover_text = [
            f"{task.feature_name}: {label}<br>value: {num:.1f}"
            for label, num in zip(labels_e, numeric_e)
        ]

        fig.add_trace(
            go.Scatter3d(
                x=z_e[:, 0],
                y=z_e[:, 1],
                z=z_e[:, 2],
                mode="markers",
                marker=dict(
                    size=2.5,
                    opacity=0.7,
                    color=numeric_e,
                    colorscale=colorscale,
                    showscale=(idx == 0),
                    colorbar=dict(title=task.feature_name, x=1.02)
                    if idx == 0
                    else None,
                ),
                name=f"Expert {expert_id}",
                text=hover_text,
                hovertemplate=(
                    f"<b>Expert {expert_id}</b><br>"
                    "z1: %{x:.3f}<br>"
                    "z2: %{y:.3f}<br>"
                    "z3: %{z:.3f}<br>"
                    "%{text}"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

        scene_name = f"scene{idx + 1}" if idx > 0 else "scene"
        fig.update_layout(
            **{
                scene_name: dict(
                    xaxis_title="z1",
                    yaxis_title="z2",
                    zaxis_title="z3",
                )
            }
        )

    fig.update_layout(
        height=400 * rows,
        width=400 * cols,
        title_text=f"SMIXAE Manifold Probing: {task.name} - {task.description}",
        showlegend=False,
    )

    filename = output_dir / f"manifold_probe_{task.name}.html"
    fig.write_html(str(filename))
    print(f"Saved: {filename}")


def plot_task_single_expert_labeled(
    task: ProbingTask,
    expert_id: int,
    fisher_score: float,
    z_np: np.ndarray,
    examples: list[ProbingExample],
    output_dir: Path,
):
    """Plot a single expert with text labels at mean positions per feature value."""
    feature_to_indices = defaultdict(list)
    for i, ex in enumerate(examples):
        feature_to_indices[ex.feature_value].append(i)

    fig = go.Figure()

    for feat_idx, value in enumerate(task.feature_values):
        indices = feature_to_indices[value]
        z_feat = z_np[indices, expert_id, :]
        norms = np.linalg.norm(z_feat, axis=-1)
        active = norms > 1e-3

        if active.sum() < 1:
            continue

        z_active = z_feat[active]
        mean_z = z_active.mean(axis=0)
        numeric = (
            task.feature_numeric[feat_idx] if task.feature_numeric else float(feat_idx)
        )

        cmin = min(task.feature_numeric)
        cmax = max(task.feature_numeric)

        # All points (small, transparent)
        fig.add_trace(
            go.Scatter3d(
                x=z_active[:, 0],
                y=z_active[:, 1],
                z=z_active[:, 2],
                mode="markers",
                marker=dict(
                    size=1.5,
                    opacity=0.3,
                    color=numeric,
                    colorscale=get_colorscale(task),
                    cmin=cmin,
                    cmax=cmax,
                    showscale=False,
                ),
                name=value,
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Mean position with label
        fig.add_trace(
            go.Scatter3d(
                x=[mean_z[0]],
                y=[mean_z[1]],
                z=[mean_z[2]],
                mode="markers+text",
                marker=dict(
                    size=6,
                    opacity=1.0,
                    color=numeric,
                    colorscale=get_colorscale(task),
                    cmin=cmin,
                    cmax=cmax,
                    showscale=(feat_idx == 0),
                    colorbar=dict(title=task.feature_name) if feat_idx == 0 else None,
                ),
                text=[value],
                textposition="top center",
                textfont=dict(size=10),
                name=value,
                showlegend=True,
                hovertemplate=f"<b>{value}</b><br>z1: %{{x:.3f}}<br>z2: %{{y:.3f}}<br>z3: %{{z:.3f}}<extra></extra>",
            )
        )

    # Draw lines connecting sequential feature values
    if len(task.feature_numeric) > 1:
        ordered_means = []
        for feat_idx, value in enumerate(task.feature_values):
            indices = feature_to_indices[value]
            z_feat = z_np[indices, expert_id, :]
            norms = np.linalg.norm(z_feat, axis=-1)
            active = norms > 1e-3
            if active.sum() >= 1:
                ordered_means.append(z_feat[active].mean(axis=0))

        if len(ordered_means) > 1:
            ordered_means = np.array(ordered_means)
            if task.cyclic:
                ordered_means = np.vstack([ordered_means, ordered_means[0:1]])

            fig.add_trace(
                go.Scatter3d(
                    x=ordered_means[:, 0],
                    y=ordered_means[:, 1],
                    z=ordered_means[:, 2],
                    mode="lines",
                    line=dict(color="rgba(100,100,100,0.5)", width=2),
                    name="path",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        title=f"Expert {expert_id}: {task.name} (Fisher: {fisher_score:.1f}) — {task.description}",
        scene=dict(xaxis_title="z1", yaxis_title="z2", zaxis_title="z3"),
        height=700,
        width=900,
    )

    filename = output_dir / f"manifold_probe_{task.name}_expert{expert_id}_labeled.html"
    fig.write_html(str(filename))
    print(f"Saved: {filename}")


# ============================================================
# 7. Summary statistics
# ============================================================
def compute_task_summary(
    task: ProbingTask,
    expert_ids: np.ndarray,
    fisher_scores: np.ndarray,
    z_np: np.ndarray,
    examples: list[ProbingExample],
) -> dict:
    feature_to_indices = defaultdict(list)
    for i, ex in enumerate(examples):
        feature_to_indices[ex.feature_value].append(i)

    summaries = []
    for expert_id in expert_ids:
        norms = np.linalg.norm(z_np[:, expert_id, :], axis=-1)
        active_frac = (norms > 1e-3).mean()

        mean_positions = []
        active_values = []
        for value in task.feature_values:
            indices = feature_to_indices[value]
            z_feat = z_np[indices, expert_id, :]
            active_mask = np.linalg.norm(z_feat, axis=-1) > 1e-3
            if active_mask.sum() > 0:
                mean_positions.append(z_feat[active_mask].mean(axis=0))
                active_values.append(value)

        if len(mean_positions) < 3:
            continue

        mean_positions = np.array(mean_positions)

        # Intrinsic dimensionality via PCA
        centered = mean_positions - mean_positions.mean(axis=0)
        if len(centered) > 1:
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)[::-1]
            total_var = eigenvalues.sum()
            if total_var > 0:
                explained = eigenvalues / total_var
                intrinsic_dim = int((explained > 0.05).sum())
            else:
                intrinsic_dim = 0
                explained = np.zeros(3)
        else:
            intrinsic_dim = 0
            explained = np.zeros(3)

        # Cyclic score: is end close to start relative to mean step?
        cyclic_score = 0.0
        if task.cyclic and len(mean_positions) >= 4:
            step_distances = np.linalg.norm(np.diff(mean_positions, axis=0), axis=-1)
            mean_step = step_distances.mean()
            close_distance = np.linalg.norm(mean_positions[-1] - mean_positions[0])
            if mean_step > 0:
                # Ratio: close_distance / mean_step. <1 means cyclic.
                cyclic_score = 1.0 - min(close_distance / (mean_step * 2), 1.0)

        summaries.append(
            {
                "expert_id": int(expert_id),
                "fisher_score": float(fisher_scores[expert_id]),
                "activation_freq": float(active_frac),
                "mean_norm": float(norms[norms > 1e-3].mean())
                if (norms > 1e-3).any()
                else 0,
                "intrinsic_dim": intrinsic_dim,
                "pca_explained_variance": [float(e) for e in explained[:3]],
                "n_feature_values_active": len(mean_positions),
                "cyclic_score": cyclic_score,
            }
        )

    return {
        "task": task.name,
        "description": task.description,
        "n_feature_values": len(task.feature_values),
        "n_templates": len(task.templates),
        "n_examples": len(examples),
        "cyclic": task.cyclic,
        "expert_summaries": summaries,
    }


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Manifold Probing for SMIXAE (Fisher Score)")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)

    # Load SAE
    print("Loading SAE...")
    sae = SAE.load_from_disk(path=CHECKPOINT_PATH, device=DEVICE)
    print(f"SAE: {sae.cfg.n_experts} experts, d_bottleneck={sae.cfg.d_bottleneck}")

    # Build probing tasks
    tasks = build_probing_tasks()
    print(f"\nBuilt {len(tasks)} probing tasks:")
    for task in tasks:
        n_examples = len(task.templates) * len(task.feature_values)
        print(
            f"  {task.name}: {len(task.feature_values)} values x "
            f"{len(task.templates)} templates = {n_examples} examples"
        )

    all_summaries = []

    for task in tasks:
        print(f"\n{'=' * 60}")
        print(f"Processing: {task.name} - {task.description}")
        print(f"{'=' * 60}")

        # Generate examples
        examples = generate_examples(task, model.tokenizer)
        print(f"Generated {len(examples)} examples")

        # Collect activations
        print("Collecting activations...")
        activations = collect_activations(model, examples, HOOK_NAME)
        print(f"Activations shape: {activations.shape}")

        # Run through SAE
        print("Encoding through SAE...")
        z_np = get_bottleneck_coords(sae, activations)
        print(f"Bottleneck coords shape: {z_np.shape}")

        # Compute expert stats
        expert_norms = np.linalg.norm(z_np, axis=-1)
        activation_freq = (expert_norms > 1e-3).mean(axis=0)
        mean_norm_per_expert = expert_norms.mean(axis=0)

        # Find responsive experts via Fisher score
        print("Computing Fisher scores...")
        top_experts, fisher_scores = find_responsive_experts(
            z_np, examples, task, n_experts=N_EXPERTS_TO_PLOT
        )

        print("Top 5 experts by Fisher score:")
        for eid in top_experts[:5]:
            print(
                f"  Expert {eid}: Fisher={fisher_scores[eid]:.2f}, "
                f"freq={activation_freq[eid]:.2%}"
            )

        # Plot grid
        print("Plotting expert grid...")
        plot_task_experts(
            task=task,
            expert_ids=top_experts,
            fisher_scores=fisher_scores,
            z_np=z_np,
            examples=examples,
            activation_freq=activation_freq,
            mean_norm_per_expert=mean_norm_per_expert,
            output_dir=OUTPUT_DIR,
        )

        # Plot top 3 with labels
        for eid in top_experts[:3]:
            plot_task_single_expert_labeled(
                task=task,
                expert_id=eid,
                fisher_score=fisher_scores[eid],
                z_np=z_np,
                examples=examples,
                output_dir=OUTPUT_DIR,
            )

        # Compute summary
        summary = compute_task_summary(task, top_experts, fisher_scores, z_np, examples)
        all_summaries.append(summary)

        # Print quick stats
        print(f"\nTop 5 experts for {task.name}:")
        for s in summary["expert_summaries"][:5]:
            print(
                f"  Expert {s['expert_id']}: Fisher={s['fisher_score']:.1f}, "
                f"freq={s['activation_freq']:.2%}, "
                f"dim={s['intrinsic_dim']}, "
                f"PCA={[f'{v:.2f}' for v in s['pca_explained_variance']]}"
                + (f", cyclic={s['cyclic_score']:.2f}" if task.cyclic else "")
            )

    # Save summaries
    summary_file = OUTPUT_DIR / "probing_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved summary: {summary_file}")

    # Print global overview
    print(f"\n{'=' * 60}")
    print("GLOBAL OVERVIEW")
    print(f"{'=' * 60}")
    for summary in all_summaries:
        top = summary["expert_summaries"][0] if summary["expert_summaries"] else None
        if top:
            print(
                f"  {summary['task']:20s}  best Fisher={top['fisher_score']:8.1f}  "
                f"dim={top['intrinsic_dim']}  "
                + (f"cyclic={top['cyclic_score']:.2f}" if summary["cyclic"] else "")
            )

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()

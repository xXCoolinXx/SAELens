import random

import pandas as pd


def generate_scientific_dataset(n_samples=1000):
    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    # Placeholder names for personalization/variety
    names = [
        "Alice",
        "Bob",
        "Charlie",
        "Dr. Smith",
        "Sarah",
        "James",
        "Elena",
        "Prof. Wang",
        "the PI",
        "the technician",
    ]

    # 30 Weekday Templates
    templates = [
        "{name} will finalize the PCR results on {day}",
        "The shipment of reagents is scheduled for {day}",
        "Please ensure the incubator is cleaned by {day}",
        "The weekly lab sync has been moved to {day}",
        "Data backup must be completed every {day}",
        "The clinical trial phase begins this {day}",
        "Is the centrifuge maintenance occurring on {day}",
        "{name} is presenting the literature review on {day}",
        "We expect the peer review feedback by {day}",
        "The microscope calibration is due on {day}",
        "Submit the grant proposal before {day}",
        "The liquid nitrogen refill happens on {day}",
        "The sample size will be recalculated on {day}",
        "Our collaborator is visiting the facility on {day}",
        "The ethics committee will meet on {day}",
        "Please verify the freezer temperature logs for {day}",
        "{name} scheduled the MRI scan for {day}",
        "The chemical waste pickup is every {day}",
        "We will start the sequencing run on {day}",
        "The department seminar is hosted on {day}",
        "Check the titration levels again on {day}",
        "The statistical analysis will be performed on {day}",
        "Are we still on track for the launch this {day}",
        "{name} noted a discrepancy in the logs from {day}",
        "The autoclave is out of service until {day}",
        "The final abstract is due this {day}",
        "Re-calibrate the mass spectrometer on {day}",
        "The longitudinal study concludes on {day}",
        "The lab will be closed for the holiday on {day}",
        "{name} is responsible for the morning rounds on {day}",
    ]

    data = []
    for _ in range(n_samples):
        day = random.choice(days)
        name = random.choice(names)
        template = random.choice(templates)

        # Fill in both name and day
        sentence = template.format(name=name, day=day)
        data.append({"Sentence": sentence, "Label": day})

    return pd.DataFrame(data)


# Generate 1,000 samples
df_1k = generate_scientific_dataset(1000)

df_1k.to_csv("weekdays.csv")

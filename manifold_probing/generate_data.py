import random

import pandas as pd

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


def generate_weekdays(n_samples=1000):
    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
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
        data.append({"Sentence": sentence, "Label": f"{days.index(day)}_{day}"})

    return pd.DataFrame(data)


def generate_hours(n_samples=1000):
    # Generate a list of hours (01:00 to 12:00 and 13:00 to 00:00)
    # Includes both AM/PM and 24h formats for better variety
    hours = [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
    ]

    templates = [
        "{name} set the alarm for {hour}",
        "The incubation period ends at {hour}",
        "Please check the sample integrity at {hour}",
        "The automated sequence is programmed for {hour}",
        "Meeting with the ethics board starts at {hour}",
        "Data synchronization will occur at {hour}",
        "{name} will start the titration at {hour}",
        "The lab access logs recorded an entry at {hour}",
        "Pressure readings must be logged at {hour}",
        "The cooling system cycles off at {hour}",
        "Expect the delivery of isotopes by {hour}",
        "The centrifuge run finishes at {hour}",
        "{name} reported a power surge around {hour}",
        "Final calibration is scheduled for {hour}",
        "The server maintenance window begins at {hour}",
        "Review the preliminary results at {hour}",
        "Shift handover is prompt at {hour}",
        "The chemical reaction reached peak at {hour}",
        "Emergency backup systems were tested at {hour}",
        "{name} will calibrate the sensors at {hour}",
        "The last observation was recorded at {hour}",
        "Please shut down the workstations by {hour}",
        "The ventilation system increases flow at {hour}",
        "Safety inspections are conducted at {hour}",
        "{name} is scheduled for bench work at {hour}",
        "The biopsy results are expected by {hour}",
        "Monitor the heart rate variability until {hour}",
        "The sterilization cycle completes at {hour}",
        "Update the project management board by {hour}",
        "{name} will upload the raw data at {hour}",
    ]

    data = []
    for _ in range(n_samples):
        hour = random.choice(hours)
        name = random.choice(names)
        template = random.choice(templates)

        sentence = template.format(name=name, hour=hour)
        data.append({"Sentence": sentence, "Label": f"{hours.index(hour)}_{hour}"})

    return pd.DataFrame(data)


# Generate 1,000 samples
df_1k = generate_weekdays(1000)
df_1k.to_csv("weekdays.csv")

df_hours = generate_hours()
df_hours.to_csv("hours.csv")

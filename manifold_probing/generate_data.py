import itertools
import random

import pandas as pd

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
    "Dr. Patel",
    "Maria",
    "the postdoc",
    "Kevin",
    "Dr. Chen",
    "Fatima",
    "the intern",
    "Prof. Müller",
    "Yuki",
    "the supervisor",
    "Raj",
    "Dr. Okafor",
    "the nurse",
    "Liam",
    "Dr. Rossi",
    "Amara",
    "the coordinator",
    "Prof. Kim",
    "Omar",
    "the analyst",
]


def _enumerate_unique(templates, labels_list, label_fmt, n_samples):
    """
    Enumerate all unique (template × name × label_value) combinations,
    deduplicate on rendered sentence, shuffle, and return up to n_samples.
    """
    combos = list(itertools.product(templates, names, labels_list))
    random.shuffle(combos)

    seen = set()
    data = []
    for template, name, label_val in combos:
        sentence = template.format(name=name, **{label_fmt[0]: label_val})
        if sentence in seen:
            continue
        seen.add(sentence)
        idx = labels_list.index(label_val)
        data.append(
            {
                "Sentence": sentence,
                "Label": f"{idx:02}_{label_val}",
            }
        )
        if len(data) >= n_samples:
            break

    random.shuffle(data)
    return pd.DataFrame(data)


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

    n_with_name = sum(1 for t in templates if "{name}" in t)
    n_without = len(templates) - n_with_name
    max_unique = n_with_name * len(names) * len(days) + n_without * len(days)
    print(
        f"Weekdays: {len(templates)} templates × {len(names)} names × "
        f"{len(days)} days → {max_unique} max unique sentences"
    )

    return _enumerate_unique(templates, days, ("day",), n_samples)


def generate_hours(n_samples=1000):
    hours = [
        "1AM",
        "2AM",
        "3AM",
        "4AM",
        "5AM",
        "6AM",
        "7AM",
        "8AM",
        "9AM",
        "10AM",
        "11AM",
        "12AM",
        "1PM",
        "2PM",
        "3PM",
        "4PM",
        "5PM",
        "6PM",
        "7PM",
        "8PM",
        "9PM",
        "10PM",
        "11PM",
        "12PM",
    ]

    templates = [
        "{name} set the alarm for {hour}",
        "The incubation period ends at the hour of {hour}",
        "Please check the sample integrity at the hour of {hour}",
        "The automated sequence is programmed for the hour of {hour}",
        "Meeting with the ethics board starts at the hour of {hour}",
        "Data synchronization will occur at the hour of {hour}",
        "{name} will start the titration at the hour of {hour}",
        "The lab access logs recorded an entry at the hour of {hour}",
        "Pressure readings must be logged at the hour of {hour}",
        "The cooling system cycles off at the hour of {hour}",
        "Expect the delivery of isotopes by the hour of {hour}",
        "The centrifuge run finishes at the hour of {hour}",
        "{name} reported a power surge around the hour of {hour}",
        "Final calibration is scheduled for the hour of {hour}",
        "The server maintenance window begins at the hour of {hour}",
        "Review the preliminary results at the hour of {hour}",
        "Shift handover is prompt, at the hour of {hour}",
        "The chemical reaction reached peak at the hour of {hour}",
        "Emergency backup systems were tested at the hour of {hour}",
        "{name} will calibrate the sensors at the hour of {hour}",
        "The last observation was recorded at the hour of {hour}",
        "Please shut down the workstations by the hour of {hour}",
        "The ventilation system increases flow at the hour of {hour}",
        "Safety inspections are conducted at the hour of {hour}",
        "{name} is scheduled for bench work at the hour of {hour}",
        "The biopsy results are expected by the hour of {hour}",
        "Monitor the heart rate variability until the hour of {hour}",
        "The sterilization cycle completes at the hour of {hour}",
        "Update the project management board by the hour of {hour}",
        "{name} will upload the raw data at the hour of {hour}",
    ]

    n_with_name = sum(1 for t in templates if "{name}" in t)
    n_without = len(templates) - n_with_name
    max_unique = n_with_name * len(names) * len(hours) + n_without * len(hours)
    print(
        f"Hours: {len(templates)} templates × {len(names)} names × "
        f"{len(hours)} hours → {max_unique} max unique sentences"
    )

    return _enumerate_unique(templates, hours, ("hour",), n_samples)


def generate_temperatures(n_samples=1000):
    temps = [
        "freezing",
        "frigid",
        "cold",
        "chilly",
        "cool",
        "mild",
        "warm",
        "hot",
        "sweltering",
        "scorching",
        "boiling",
    ]

    templates = [
        "{name} noted the sample chamber felt {temp}",
        "The lab conditions today are {temp}",
        "Outside it is absolutely {temp}",
        "{name} complained the office was {temp}",
        "The reactor core temperature reads {temp}",
        "Patients reported feeling {temp} during the trial",
        "The greenhouse environment is {temp}",
        "{name} described the cleanroom as {temp}",
        "The storage unit is running {temp}",
        "Field conditions were {temp} during collection",
        "The water bath feels {temp}",
        "{name} adjusted the thermostat because it was {temp}",
        "The incubation environment should remain {temp}",
        "Volunteers rated the testing room as {temp}",
        "The fermentation tank is currently {temp}",
        "{name} flagged that the freezer felt {temp}",
        "The climate chamber was set to {temp}",
        "Surface readings indicate conditions are {temp}",
        "The server room is dangerously {temp}",
        "{name} measured the soil temperature as {temp}",
        "Ambient conditions in the corridor are {temp}",
        "The curing oven is running {temp}",
        "{name} said the walk-in cooler felt {temp}",
        "The drying chamber atmosphere is {temp}",
        "Morning readings showed the habitat was {temp}",
        "The bioreactor jacket temperature is {temp}",
        "{name} recorded that the growth chamber felt {temp}",
        "The ventilation output is blowing {temp}",
        "The reagent shelf area felt {temp}",
        "{name} reported the autoclave room as {temp}",
    ]

    n_with_name = sum(1 for t in templates if "{name}" in t)
    n_without = len(templates) - n_with_name
    max_unique = n_with_name * len(names) * len(temps) + n_without * len(temps)
    print(
        f"Temperatures: {len(templates)} templates × {len(names)} names × "
        f"{len(temps)} temps → {max_unique} max unique sentences"
    )

    return _enumerate_unique(templates, temps, ("temp",), n_samples)


df_weekdays = generate_weekdays(1000)
df_weekdays.to_csv("weekdays.csv", index=False)
print(f"Wrote {len(df_weekdays)} unique weekday samples")

df_hours = generate_hours(1500)
df_hours.to_csv("hours.csv", index=False)
print(f"Wrote {len(df_hours)} unique hour samples")

df_temps = generate_temperatures(1000)
df_temps.to_csv("temperatures.csv", index=False)
print(f"Wrote {len(df_temps)} unique temperature samples")

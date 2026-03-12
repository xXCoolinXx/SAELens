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


def generate_time_units(n_samples=1000):
    # Ordered from shortest to longest duration
    units = [
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
    ]

    templates = [
        "{name} set the experiment timer for one {unit}",
        "The process is measured in units of one {unit}",
        "Each interval in the protocol corresponds to one {unit}",
        "The signal persists for approximately one {unit}",
        "Resolution of the sensor is roughly one {unit}",
        "{name} noted the delay was about one {unit}",
        "The reaction completes in exactly one {unit}",
        "The standard interval for this assay is one {unit}",
        "{name} calculated the half-life as roughly one {unit}",
        "The oscillation period measures one {unit}",
        "Events are logged to the nearest {unit}",
        "The clock ticks once per {unit}",
        "The gap between readings is one {unit}",
        "{name} reported latency of approximately one {unit}",
        "The simulation advances by one {unit}",
        "Phase transitions occur every {unit}",
        "Data is captured at a resolution of one {unit}",
        "The protocol requires a pause of one {unit}",
        "{name} measured the response time in one {unit}",
        "The cache refreshes every {unit}",
        "Cell division occurs on the order of one {unit}",
        "The epoch duration is fixed at one {unit}",
        "{name} confirmed the lag was under one {unit}",
        "The retention policy spans one {unit}",
        "Sample decay is tracked per {unit}",
        "The synchronization window aligns to one {unit}",
        "{name} budgeted resources by the {unit}",
        "The checkpoint interval is set to one {unit}",
        "{name} recorded the duration as one {unit}",
        "The experiment was designed around the timescale of one {unit}",
    ]

    n_with_name = sum(1 for t in templates if "{name}" in t)
    n_without = len(templates) - n_with_name
    max_unique = n_with_name * len(names) * len(units) + n_without * len(units)
    print(
        f"Time units: {len(templates)} templates × {len(names)} names × "
        f"{len(units)} units → {max_unique} max unique sentences"
    )

    return _enumerate_unique(templates, units, ("unit",), n_samples)


def generate_body_parts(n_samples=1000):
    # Ordered head-to-toe
    parts = [
        "head",
        "face",
        "neck",
        "shoulder",
        "chest",
        "back",
        "arm",
        "elbow",
        "wrist",
        "hand",
        "abdomen",
        "hip",
        "leg",
        "knee",
        "ankle",
        "foot",
    ]

    templates = [
        "The scan revealed an abnormality in the {part}",
        "{name} reported persistent soreness in the {part}",
        "The injury was localized to the {part}",
        "The nurse applied a bandage to the {part}",
        "{name} noticed visible swelling in the {part}",
        "The X-ray focused on the patient's {part}",
        "Range of motion was restricted in the {part}",
        "{name} documented bruising on the {part}",
        "The physical exam highlighted tenderness in the {part}",
        "The protective gear is designed for the {part}",
        "{name} measured the circumference of the {part}",
        "The biopsy was taken from tissue near the {part}",
        "Temperature elevation was noted at the {part}",
        "{name} reported numbness spreading from the {part}",
        "The rash first appeared on the {part}",
        "The surgeon focused the incision near the {part}",
        "{name} documented visible asymmetry in the {part}",
        "The compression sleeve is worn on the {part}",
        "Reflexes were tested at the {part}",
        "{name} reported that the discomfort originated in the {part}",
        "The burn was classified as superficial, affecting the {part}",
        "Muscle weakness was observed in the {part}",
        "{name} applied ice to reduce inflammation in the {part}",
        "The fracture was confirmed in the {part}",
        "Skin discoloration was observed on the {part}",
        "{name} noted restricted blood flow to the {part}",
        "The physical therapist worked on mobility of the {part}",
        "Lymph node enlargement was found near the {part}",
        "{name} described a tingling sensation in the {part}",
        "The imaging clearly depicted damage to the {part}",
    ]

    n_with_name = sum(1 for t in templates if "{name}" in t)
    n_without = len(templates) - n_with_name
    max_unique = n_with_name * len(names) * len(parts) + n_without * len(parts)
    print(
        f"Body parts: {len(templates)} templates × {len(names)} names × "
        f"{len(parts)} parts → {max_unique} max unique sentences"
    )

    return _enumerate_unique(templates, parts, ("part",), n_samples)


def generate_living_things(n_samples=1000):
    # Ordered from simpler to more complex organisms (plants then animals)
    organisms = [
        "moss",
        "fern",
        "grass",
        "flower",
        "shrub",
        "tree",
        "insect",
        "arachnid",
        "crustacean",
        "fish",
        "amphibian",
        "reptile",
        "bird",
        "mammal",
    ]

    templates = [
        "The researcher identified the specimen as a {organism}",
        "{name} photographed what appeared to be a {organism}",
        "The field guide confirmed the find was a {organism}",
        "Samples were collected from a living {organism}",
        "{name} documented the habitat of the {organism}",
        "The ecology report focused on the local {organism}",
        "DNA sequencing confirmed the sample came from a {organism}",
        "{name} observed the behavior of a wild {organism}",
        "The museum exhibit featured a preserved {organism}",
        "The invasive species turned out to be a {organism}",
        "{name} extracted RNA from the tissue of a {organism}",
        "The biome is dominated by the {organism}",
        "Children in the class were asked to draw a {organism}",
        "{name} trained for years on identifying a {organism}",
        "The fossil record shows evidence of the ancient {organism}",
        "Conservation efforts focused on preserving the {organism}",
        "{name} cultured a colony derived from a {organism}",
        "The biodiversity index recorded the presence of a {organism}",
        "The nature documentary featured the remarkable {organism}",
        "{name} noted that the diet study examined a {organism}",
        "The ecosystem depends heavily on the {organism}",
        "The genome was successfully sequenced from a {organism}",
        "{name} spent the summer studying a local {organism}",
        "The biology textbook chapter covered the {organism}",
        "Environmental impact was assessed for the {organism}",
        "{name} confirmed the endangered status of the {organism}",
        "The sanctuary was established to protect the {organism}",
        "Biochemists extracted compounds from the {organism}",
        "{name} tagged and released the captured {organism}",
        "The grant funded a three-year study of the {organism}",
    ]

    n_with_name = sum(1 for t in templates if "{name}" in t)
    n_without = len(templates) - n_with_name
    max_unique = n_with_name * len(names) * len(organisms) + n_without * len(organisms)
    print(
        f"Living things: {len(templates)} templates × {len(names)} names × "
        f"{len(organisms)} organisms → {max_unique} max unique sentences"
    )

    return _enumerate_unique(templates, organisms, ("organism",), n_samples)


def generate_months(n_samples=1000):
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

    templates = [
        "{name} will submit the annual report in {month}",
        "The grant deadline falls in {month}",
        "The conference is scheduled for {month}",
        "Field data collection begins in {month}",
        "{name} plans to defend the thesis in {month}",
        "The fiscal year closes at the end of {month}",
        "The cohort study enrollment opens in {month}",
        "{name} noted the equipment arrived in {month}",
        "The review board meets annually in {month}",
        "The lab renovation is planned for {month}",
        "Sample collection was completed in {month}",
        "{name} presented the interim findings in {month}",
        "The breeding season peaks in {month}",
        "The journal submission window opens in {month}",
        "The fellowship applications are due in {month}",
        "{name} recorded the highest yield in {month}",
        "The symposium takes place every {month}",
        "The pilot study wrapped up in {month}",
        "The funding cycle resets each {month}",
        "{name} returned from fieldwork in {month}",
        "The accreditation review is scheduled for {month}",
        "Animal migration peaks in {month}",
        "{name} confirmed the calibration was done in {month}",
        "The onboarding of new staff happens every {month}",
        "The protocol amendment was approved in {month}",
        "{name} noticed the anomaly first appeared in {month}",
        "The seasonal survey is conducted each {month}",
        "The lab will be closed for two weeks in {month}",
        "The reagent stock is replenished every {month}",
        "{name} finalized the analysis in {month}",
    ]

    n_with_name = sum(1 for t in templates if "{name}" in t)
    n_without = len(templates) - n_with_name
    max_unique = n_with_name * len(names) * len(months) + n_without * len(months)
    print(
        f"Months: {len(templates)} templates × {len(names)} names × "
        f"{len(months)} months → {max_unique} max unique sentences"
    )

    return _enumerate_unique(templates, months, ("month",), n_samples)


def main():
    df_weekdays = generate_weekdays(1000)
    df_weekdays.to_csv("weekdays.csv", index=False)
    print(f"Wrote {len(df_weekdays)} unique weekday samples\n")

    df_hours = generate_hours(1500)
    df_hours.to_csv("hours.csv", index=False)
    print(f"Wrote {len(df_hours)} unique hour samples\n")

    df_temps = generate_temperatures(1000)
    df_temps.to_csv("temperatures.csv", index=False)
    print(f"Wrote {len(df_temps)} unique temperature samples\n")

    df_time_units = generate_time_units(1000)
    df_time_units.to_csv("time_units.csv", index=False)
    print(f"Wrote {len(df_time_units)} unique time unit samples\n")

    df_body_parts = generate_body_parts(1000)
    df_body_parts.to_csv("body_parts.csv", index=False)
    print(f"Wrote {len(df_body_parts)} unique body part samples\n")

    df_living_things = generate_living_things(1000)
    df_living_things.to_csv("living_things.csv", index=False)
    print(f"Wrote {len(df_living_things)} unique living things samples\n")

    df_months = generate_months(1000)
    df_months.to_csv("months.csv", index=False)
    print(f"Wrote {len(df_months)} unique month samples\n")


if __name__ == "__main__":
    main()

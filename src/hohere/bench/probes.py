"""Minimal-pair probes: does the model's answer depend on what the user actually said?

Each pair has two questions that differ in one key word (or one key fact) and the two
matching answers. A model that listens finds answer X more likely after question X than
after question Y. Texts are built from templates here, never by the dialogue generator,
so they are not drawn from the training distribution of hohere dialogues.

kind = "echo":  the answer repeats the differing word ("the red cup" -> "The red cup is...").
kind = "infer": the answer follows from the meaning without repeating it
                ("It's pouring outside" -> "Yes, take the umbrella").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# (question template, answer template); {x} is the slot, {X} the capitalized slot.
ECHO: dict[str, tuple[list[str], list[tuple[str, str]]]] = {
    "color": (
        ["red", "blue", "green", "yellow", "black", "white", "orange", "purple", "pink",
         "brown", "gray"],
        [
            ("Can you find the {x} mug for me?", "The {x} mug is on the top shelf."),
            ("I want to paint the bedroom {x}.", "{X} would look great in the bedroom."),
            ("Do you have this shirt in {x}?", "Yes, we have it in {x}."),
            ("My car is the {x} one by the gate.", "Got it, the {x} car by the gate."),
            ("I lost a {x} umbrella yesterday.", "A {x} umbrella, let me check lost and found."),
            ("Pass me the {x} pen, please.", "Here's the {x} pen."),
        ],
    ),
    "number": (
        ["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
         "twelve"],
        [
            ("I need {x} tickets for tonight.", "Okay, {x} tickets for tonight."),
            ("Set a timer for {x} minutes.", "Timer set for {x} minutes."),
            ("We're a group of {x} people.", "A table for {x}, no problem."),
            ("I'll take {x} apples, please.", "Sure, {x} apples."),
            ("My flight leaves in {x} hours.", "So you have {x} hours, plenty of time."),
            ("The meeting is on floor {x}.", "Floor {x}, the elevator is on your left."),
        ],
    ),
    "name": (
        ["Anna", "David", "Maria", "James", "Sophie", "Daniel", "Laura", "Peter", "Emma",
         "Michael", "Olivia", "Thomas"],
        [
            ("Please call {x} for me.", "Calling {x} now."),
            ("I'm meeting {x} for lunch.", "Say hi to {x} for me."),
            ("Did {x} send the report?", "Yes, {x} sent it this morning."),
            ("Remind me to text {x} later.", "Okay, I'll remind you to text {x}."),
            ("{X} is coming to dinner tonight.", "Great, I'll set a place for {x}."),
        ],
    ),
    "city": (
        ["Paris", "Tokyo", "Berlin", "Madrid", "Chicago", "Sydney", "Toronto", "Rome",
         "Dublin", "Boston", "Lisbon", "Vienna"],
        [
            ("Book me a flight to {x}.", "Looking for flights to {x}."),
            ("What's the weather like in {x}?", "Let me check the weather in {x}."),
            ("My sister lives in {x}.", "Oh, {x} is a lovely city."),
            ("I'm moving to {x} next month.", "Moving to {x}, how exciting."),
            ("How far is {x} from here?", "Let me check the distance to {x}."),
        ],
    ),
    "object": (
        ["umbrella", "wallet", "laptop", "jacket", "passport", "charger", "backpack",
         "notebook", "scarf", "phone", "camera", "helmet"],
        [
            ("I can't find my {x}.", "Your {x} is on the table by the door."),
            ("Did I pack my {x}?", "Yes, your {x} is in the side pocket."),
            ("I left my {x} at the office.", "You can pick up your {x} tomorrow."),
            ("Remind me to bring my {x}.", "Okay, I'll remind you to bring your {x}."),
            ("I need a new {x}.", "Let's look for a new {x}."),
        ],
    ),
    "weekday": (
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        [
            ("Can we meet on {x}?", "Sure, {x} works for me."),
            ("The party is on {x}.", "Okay, the party is on {x}."),
            ("My dentist appointment is on {x}.", "Got it, dentist on {x}."),
            ("Move the meeting to {x}.", "Done, the meeting is now on {x}."),
            ("I'm off work on {x}.", "Nice, you're free on {x}."),
        ],
    ),
}

# Each frame: (category, [(question_x, question_y), ...], answer_x, answer_y).
# The question variants are paraphrases; x and y differ in the fact that decides the answer.
INFER: list[tuple[str, list[tuple[str, str]], str, str]] = [
    ("weather", [
        ("It's pouring outside, should I take an umbrella?",
         "It's sunny outside, should I take an umbrella?"),
        ("It's raining hard right now, do I need an umbrella?",
         "It's bright and dry right now, do I need an umbrella?"),
        ("Look at that storm, will I need an umbrella?",
         "Look at that clear sky, will I need an umbrella?"),
    ], "Yes, take the umbrella, you'll need it.", "No, you won't need an umbrella today."),
    ("temperature", [
        ("It's freezing in here.", "It's boiling in here."),
        ("I'm so cold in this room.", "I'm so hot in this room."),
        ("Brr, this room is icy.", "Phew, this room is stuffy and hot."),
    ], "Let me turn the heating up.", "Let me turn the air conditioning on."),
    ("greeting", [
        ("Good morning!", "Good night!"),
        ("Morning, I just woke up.", "Night, I'm off to bed."),
        ("Hi, I'm just getting up.", "Okay, I'm going to sleep now."),
    ], "Good morning, did you sleep well?", "Good night, sleep well."),
    ("hunger", [
        ("I'm starving.", "I'm so full."),
        ("I haven't eaten all day.", "I just ate a huge lunch."),
        ("My stomach is growling.", "I couldn't eat another bite."),
    ], "Let's get you something to eat.", "Then let's skip dessert."),
    ("light", [
        ("It's too dark in here.", "It's too bright in here."),
        ("I can barely see anything in this room.", "The light in here is blinding."),
        ("This room is so dim.", "This room is so glaring."),
    ], "I'll turn the lights on.", "I'll dim the lights."),
    ("time", [
        ("I'm running late.", "I'm way too early."),
        ("The bus leaves in two minutes and I'm still home.",
         "The bus leaves in an hour and I'm already at the stop."),
        ("Oh no, I overslept.", "I woke up two hours before my alarm."),
    ], "Hurry, you can still make it.", "Then you have time for a coffee."),
    ("news", [
        ("I just got some great news!", "I just got some terrible news."),
        ("I passed my driving test!", "I failed my driving test."),
        ("I got the job!", "I didn't get the job."),
    ], "That's wonderful, congratulations!", "Oh no, I'm so sorry to hear that."),
    ("volume", [
        ("I can't hear the music at all.", "The music is way too loud."),
        ("Can you make it louder? It's so quiet.", "Can you make it quieter? It hurts my ears."),
        ("I can barely hear the radio.", "The radio is blasting."),
    ], "Sure, turning it up.", "Sure, turning it down."),
    ("health", [
        ("I feel great today.", "I feel sick today."),
        ("I slept well and I'm full of energy.", "I have a fever and a sore throat."),
        ("I'm feeling fantastic this morning.", "I'm feeling awful this morning."),
    ], "Glad to hear it, enjoy your day.", "Sorry to hear that, maybe rest at home."),
    ("thirst", [
        ("I'm really thirsty.", "I've had enough to drink."),
        ("My mouth is so dry.", "I just drank three glasses of water."),
        ("I need something to drink.", "No more drinks for me, thanks."),
    ], "Here, have a glass of water.", "Okay, I won't pour you another one."),
]


@dataclass(frozen=True)
class Probe:
    pair_id: str
    kind: str  # "echo" | "infer"
    category: str
    question_x: str
    question_y: str
    answer_x: str
    answer_y: str
    slot_x: str | None = None
    slot_y: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _fill(template: str, x: str) -> str:
    return template.replace("{X}", x[:1].upper() + x[1:]).replace("{x}", x)


def build_probes(pairs_per_echo_category: int = 70, seed: int = 2026) -> list[Probe]:
    """Deterministic probe set: echo pairs per category plus every infer pairing.

    Infer frames pair each x paraphrase with each y paraphrase (3 x 3 = 9 pairs per frame).
    """
    rng = np.random.default_rng(seed)
    probes: list[Probe] = []
    for category, (values, templates) in ECHO.items():
        combos = [
            (t, i, j)
            for t in range(len(templates))
            for i in range(len(values))
            for j in range(i + 1, len(values))
        ]
        if pairs_per_echo_category > len(combos):
            raise ValueError(f"{category}: only {len(combos)} distinct pairs")
        chosen = rng.choice(len(combos), size=pairs_per_echo_category, replace=False)
        for n, k in enumerate(sorted(chosen)):
            t, i, j = combos[k]
            x, y = values[i], values[j]
            if rng.random() < 0.5:
                x, y = y, x
            q, a = templates[t]
            probes.append(
                Probe(
                    pair_id=f"echo-{category}-{n:03d}",
                    kind="echo",
                    category=category,
                    question_x=_fill(q, x),
                    question_y=_fill(q, y),
                    answer_x=_fill(a, x),
                    answer_y=_fill(a, y),
                    slot_x=x,
                    slot_y=y,
                )
            )
    for category, variants, ans_x, ans_y in INFER:
        n = 0
        for qx, _ in variants:
            for _, qy in variants:
                probes.append(
                    Probe(
                        pair_id=f"infer-{category}-{n:03d}",
                        kind="infer",
                        category=category,
                        question_x=qx,
                        question_y=qy,
                        answer_x=ans_x,
                        answer_y=ans_y,
                    )
                )
                n += 1
    return probes

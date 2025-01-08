WIDTH: int = 500
HEIGHT: int = 700

ENTRY_WIDTH: int = int(WIDTH / 10)
BUTTON_WIDTH: int = ENTRY_WIDTH - 5

BG_COLOR: str = "#191919"
FG_COLOR: str = "#FFFFFF"
HL_COLOR: str = "#017300"
BLACK: str = "#000000"
GREY: str = "#808080"
HL_BG_COLOR: str = "#0000C5"

FONT = lambda size: ("Arial", size)
LABEL_FONT_SIZE: int = 13

ENTRY_PADY: int = 5
ENTRY_PADX: int = 5
LABEL_PADY: int = 5

SELECTION_STRATEGY = ["Selection strategy: TOURNAMENT", "Selection strategy: ROULETTE", "Selection strategy: BEST"]
CROSS_STRATEGY = ["Cross strategy: LINEAR", "Cross strategy: ALFA", "Cross strategy: ALFA_BETA",
                  "Cross strategy: ARITHMETIC", "Cross strategy: MEAN"]
MUTATION_STRATEGY = ["Mutation strategy: UNIFORM", "Mutation strategy: GAUSSIAN"]
STRATEGY = lambda x: x.split(':')[1].strip()

FUNCTIONS_LIST = ['Function: ROSENBROCK', 'Function: HAPPYCAT']
FUNC = lambda x: x.split(':')[1].strip()
VALIDATE = lambda x: x if x > 0 else None

DEFAULTS = {
    "Interval start (a)": "-2.048",
    "Interval end (b)": "2.048",
    "Population size": "200",
    "Number of variables": "3",
    "Number of generations": "389",
    "Number of individuals (best / tournament)": "3",
    "Number of individuals (elite)": "6",
    "Cross probability": "0.8925",
    "Mutation probability": "0.1689"
}

from tkinter import *
from tkinter.ttk import Style
from tkinter import ttk

import numpy as np

from source.gui.widget_builder import WidgetBuilder
from source.gui.common import *

class Application:
    def __init__(self) -> None:
        root = Tk()

        style = Style()
        style.theme_use('clam')
        style.configure("TCombobox", fieldbackground=GREY, background=GREY, font=FONT(LABEL_FONT_SIZE))
        style.configure('TLabel', font=FONT(LABEL_FONT_SIZE), background=BG_COLOR, foreground=FG_COLOR)
        style.configure('TEntry', font=FONT(LABEL_FONT_SIZE), background=FG_COLOR, foreground=BLACK, width=ENTRY_WIDTH)
        style.configure('TCheckbutton', font=(FONT(LABEL_FONT_SIZE)))

        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()
        ws = (ws / 2) - (WIDTH / 2)
        hs = (hs / 2) - (HEIGHT / 2) - 25

        root.configure(background=BG_COLOR)
        root.geometry("%dx%d+%d+%d" % (WIDTH, HEIGHT, ws, hs))
        root.title("Classical genetic algorithm")
        root.bind("<Escape>", lambda e: e.widget.quit())
        root.focus_set()
        root.resizable(False, False)
        self.root = root

        frame = Frame(root)
        frame.configure({"background": BG_COLOR})
        frame.pack(padx=ENTRY_PADX, pady=ENTRY_PADY)

        title = ttk.Label(frame, text="Configuration")
        title.configure(font=FONT(14))
        title.grid(row=0, column=0, columnspan=2)

        lbl_range_start, self.range_start = WidgetBuilder.create_row(frame, "Interval start (a)", 1, float)
        lbl_range_end, self.range_end = WidgetBuilder.create_row(frame, "Interval end (b)", 2, float)
        lbl_population, self.population_count = WidgetBuilder.create_row(frame, "Population size", 3)
        lbl_variables, self.number_of_variables = WidgetBuilder.create_row(frame, "Number of variables", 4)
        # lbl_bits, self.bits_count = WidgetBuilder.create_row(frame, "Bits per chromosome (precision)", 5)
        lbl_epochs, self.epoch_count = WidgetBuilder.create_row(frame, "Number of generations", 6)
        lbl_best, self.individuals_best = WidgetBuilder.create_row(frame, "Number of individuals (best / tournament)", 7)
        lbl_elite, self.individuals_elite = WidgetBuilder.create_row(frame, "Number of individuals (elite)", 8)
        # lbl_intersection, self.intersection_number = WidgetBuilder.create_row(frame, "Number of intersections (point crossover)", 9)

        lbl_cross, self.cross_prob = WidgetBuilder.create_row(frame, "Cross probability", 10, float)
        lbl_mutation, self.mutation_prob = WidgetBuilder.create_row(frame, "Mutation probability", 11, float)
        # lbl_inversion, self.inversion_prob = WidgetBuilder.create_row(frame, "Inversion probability", 12, float)

        self.selection_var = StringVar()
        self.cross_var = StringVar()
        self.mutation_var = StringVar()
        self.function_var = StringVar()
        self.selection_strategy = WidgetBuilder.create_combobox(frame, SELECTION_STRATEGY, self.selection_var, 13)
        self.cross_method = WidgetBuilder.create_combobox(frame, CROSS_STRATEGY, self.cross_var, 14)
        self.mutation_method = WidgetBuilder.create_combobox(frame, MUTATION_STRATEGY, self.mutation_var, 15)
        self.functions = WidgetBuilder.create_combobox(frame, FUNCTIONS_LIST, self.function_var, 16)

        self.is_minimize_var = BooleanVar()
        self.is_minimize = WidgetBuilder.create_checkbox(frame, "Find minimum of the function", self.is_minimize_var, 17)

        self.button_start = WidgetBuilder.create_button(self.root, "Start")

    def start(self) -> None:
        self.root.mainloop()

    def get_algorithm_config(self) -> dict:
        res = dict()
        res['start_interval'] = self.range_start.get_value
        res['end_interval'] = self.range_end.get_value
        res['population_size'] = VALIDATE(self.population_count.get_value)
        res['number_of_variables'] = VALIDATE(self.number_of_variables.get_value)
        # res['bits_count'] = VALIDATE(self.bits_count.get_value)
        res['generations'] = VALIDATE(self.epoch_count.get_value)
        res['individuals_best'] = VALIDATE(self.individuals_best.get_value)
        res['individuals_elite'] = VALIDATE(self.individuals_elite.get_value)
        # res['intersection_number'] = VALIDATE(self.intersection_number.get_value)
        res['cross_prob'] = VALIDATE(self.cross_prob.get_value)
        res['mutation_prob'] = VALIDATE(self.mutation_prob.get_value)
        # res['inversion_prob'] = VALIDATE(self.inversion_prob.get_value)
        res['selection_strategy'] = STRATEGY(self.selection_var.get())
        res['cross_strategy'] = STRATEGY(self.cross_var.get())
        res['mutation_strategy'] = STRATEGY(self.mutation_var.get())
        res['function_name'] = FUNC(self.function_var.get())
        res['search_minimum'] = self.is_minimize_var.get()
        return res

    def popup(self, time_elapsed : float, value : float, representation : np.ndarray) -> None:
        popup = Toplevel(self.root)
        popup.configure(background=BG_COLOR)
        h = HEIGHT // 4
        w = WIDTH * 1.5
        ws = (self.root.winfo_screenwidth() / 2) - (w / 2)
        hs = (self.root.winfo_screenheight() / 2) - (h / 2)
        popup.geometry("%dx%d+%d+%d" % (w, h, ws, hs))
        popup.title("Result")
        popup.resizable(False, False)

        l1 = Label(popup, text=f"F(x) = {value}", font=FONT(LABEL_FONT_SIZE))
        l1.configure({"background": BG_COLOR, "foreground": FG_COLOR})
        l1.pack(pady=5)
        tmp = representation if len(representation[0])*len(representation) <= 10 else "[...]"
        print(f"Values = {representation}")
        l3 = Label(popup, text=f"Values = {tmp}", font=FONT(LABEL_FONT_SIZE))
        l3.configure({"background": BG_COLOR, "foreground": FG_COLOR})
        l3.pack(pady=5)
        l2 = Label(popup, text=f"Time = {time_elapsed:.2f} sec", font=FONT(LABEL_FONT_SIZE))
        l2.pack(pady=5)
        l2.configure({"background": BG_COLOR, "foreground": FG_COLOR})
        button = WidgetBuilder.create_button(popup, "OK")
        button.config(command=popup.destroy)
        button.focus_set()
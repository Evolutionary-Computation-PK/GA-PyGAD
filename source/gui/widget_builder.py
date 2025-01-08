from typing import Tuple

from source.gui.common import (ENTRY_PADY, FONT, FG_COLOR, BLACK, BG_COLOR,
                               DEFAULTS, LABEL_FONT_SIZE,
                               BUTTON_WIDTH)
from source.gui.entry_ext import EntryExt
from tkinter import Misc, Button, StringVar, BooleanVar, Checkbutton
from tkinter import ttk


class WidgetBuilder:
    @staticmethod
    def create_row(master: Misc, text: str, row: int, entry_type: type = int) -> Tuple[ttk.Label, EntryExt]:
        label = ttk.Label(master, text=text)
        label.grid(row=row, column=0, pady=ENTRY_PADY, sticky="w")
        entry = EntryExt(master, DEFAULTS[text], row, entry_type)
        return label, entry

    @staticmethod
    def create_checkbox(master : Misc, text : str, checkbox_var : BooleanVar, row: int) -> ttk.Checkbutton:
        checkbox = Checkbutton(master,
                         text=text,
                         variable=checkbox_var,
                         onvalue=True, offvalue=False)
        checkbox.grid(row=row, column=0, columnspan=2, pady=ENTRY_PADY, sticky="we")
        checkbox.configure({
            "background": BG_COLOR,
            "foreground": FG_COLOR,
            "selectcolor": BG_COLOR,
            "activebackground": BG_COLOR,
            "activeforeground": FG_COLOR
        })
        return checkbox

    @staticmethod
    def create_combobox(master : Misc, values : list, value : StringVar, row: int) -> ttk.Combobox:
        combobox = ttk.Combobox(master,
                         textvariable=value)
        combobox.configure({"background": FG_COLOR, "foreground": BLACK})
        combobox['values'] = values
        combobox['state'] = "readonly"
        combobox.current(0)
        combobox.grid(row=row, column=0, columnspan=2, pady=ENTRY_PADY, ipady=2, sticky="we")
        return combobox

    @staticmethod
    def create_button(master : Misc, text : str) -> Button:
        button = Button(master, text=text,
                        font=FONT(LABEL_FONT_SIZE),
                        width=BUTTON_WIDTH)
        button.pack(pady=ENTRY_PADY)
        return button
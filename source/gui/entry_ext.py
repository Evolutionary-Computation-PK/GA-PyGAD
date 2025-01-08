from tkinter import *
from source.gui.common import *
from tkinter import ttk

class EntryExt(ttk.Entry):
    def __init__(self, master: Misc, text: str, row: int, entry_type: type = int) -> None:
        super().__init__(master)
        self.__type = entry_type
        self.insert(0, text)
        self.modified = False
        self.bind("<Key>", self.__first_time_entry)
        self.grid(row=row, column=1, padx=ENTRY_PADX, pady=ENTRY_PADY, ipady=2, sticky="e")

    def __first_time_entry(self, event : Event) -> None:
        if event.keysym == "Tab":
            return
        if event.widget == self and not self.modified:
            self.delete(0, END)
            self.modified = True

    @property
    def get_value(self) -> float | int | None:
        value = 0
        try:
            value = self.__type(self.get())
        except ValueError:
            self.focus()
            value = None
        finally:
            return value
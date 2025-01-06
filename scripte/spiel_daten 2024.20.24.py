# %%

# %%
# directory where this script/file is saved
script_dir = os.path.dirname(os.path.abspath(__file__))
# Set the current working directory to the script directory
os.chdir(script_dir)
# %%
import pandas as pd
import numpy as np

# Anzahl der Datenpunkte (z.B. Frames oder Zeitpunkte)
n = 1000  # Du kannst die Anzahl der Datenpunkte anpassen

# Erstelle fiktive Daten für 2 Mäuse mit zufälligen x/y-Positionen
np.random.seed(42)  # Für reproduzierbare Ergebnisse

# Zufällige Positionen für Maus 1 (x/y Koordinaten)
maus1_nase_x = np.random.uniform(0, 100, n)
maus1_nase_y = np.random.uniform(0, 100, n)
maus1_hals_x = np.random.uniform(0, 100, n)
maus1_hals_y = np.random.uniform(0, 100, n)

# Zufällige Positionen für Maus 2 (x/y Koordinaten)
maus2_nase_x = np.random.uniform(0, 100, n)
maus2_nase_y = np.random.uniform(0, 100, n)
maus2_hals_x = np.random.uniform(0, 100, n)
maus2_hals_y = np.random.uniform(0, 100, n)

# Erstelle einen DataFrame mit diesen fiktiven Daten
df = pd.DataFrame(
    {
        "Frame": np.arange(1, n + 1),  # Frame Nummer
        "Maus1_Nase_X": maus1_nase_x,
        "Maus1_Nase_Y": maus1_nase_y,
        "Maus1_Hals_X": maus1_hals_x,
        "Maus1_Hals_Y": maus1_hals_y,
        "Maus2_Nase_X": maus2_nase_x,
        "Maus2_Nase_Y": maus2_nase_y,
        "Maus2_Hals_X": maus2_hals_x,
        "Maus2_Hals_Y": maus2_hals_y,
    }
)

# Zeige die ersten paar Zeilen des fiktiven Datensatzes
print(df.head())

# Exportiere den Datensatz in eine CSV-Datei (optional)
df.to_csv("fiktiver_maus_datensatz.csv", index=False)

# %%
import numpy as np


def berechne_vektor_laenge(x_werte, y_werte):
    """
    Berechnet die Länge des Vektors von jedem Punkt (x, y) zum Ursprung (0, 0).

    Parameter:
    x_werte (numpy array): X-Koordinaten
    y_werte (numpy array): Y-Koordinaten

    Rückgabe:
    numpy array: Vektor-Längen für jede Position
    """
    # Berechnung der Vektor-Länge für jede Koordinate (Pythagoras)
    laengen = np.sqrt(x_werte**2 + y_werte**2)
    return laengen


# Beispiel: Koordinaten der Nase von Maus 1 aus dem fiktiven Datensatz
maus1_nase_x = df["Maus1_Nase_X"].values  # X-Koordinaten der Nase
maus1_nase_y = df["Maus1_Nase_Y"].values  # Y-Koordinaten der Nase

# Berechne die Länge des Vektors von (0, 0) zu den x/y-Punkten der Nase
vektor_langen = berechne_vektor_laenge(maus1_nase_x, maus1_nase_y)

# Füge die Längen in den DataFrame ein (optional)
df["Maus1_Nase_Vektor_Laenge"] = vektor_langen

# Ausgabe der ersten paar berechneten Vektor-Längen
print(df[["Maus1_Nase_X", "Maus1_Nase_Y", "Maus1_Nase_Vektor_Laenge"]].head())

# %%
import numpy as np
import pandas as pd


def berechne_vektor_laenge(x_werte, y_werte):
    """
    Berechnet die Länge des Vektors von jedem Punkt (x, y) zum Ursprung (0, 0).

    Parameter:
    x_werte (numpy array): X-Koordinaten
    y_werte (numpy array): Y-Koordinaten

    Rückgabe:
    numpy array: Vektor-Längen für jede Position
    """
    laengen = np.sqrt(x_werte**2 + y_werte**2)
    return laengen


def berechne_distanz(x1, y1, x2, y2):
    """
    Berechnet die Distanz zwischen zwei Punkten (x1, y1) und (x2, y2).

    Parameter:
    x1, y1: Koordinaten des ersten Punktes (Maus 1)
    x2, y2: Koordinaten des zweiten Punktes (Maus 2)

    Rückgabe:
    numpy array: Distanzen zwischen den beiden Punkten für jede Frame
    """
    distanzen = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distanzen


# Beispiel: Koordinaten der Nase von Maus 1 und Maus 2 aus dem fiktiven Datensatz
maus1_nase_x = df["Maus1_Nase_X"].values
maus1_nase_y = df["Maus1_Nase_Y"].values
maus2_nase_x = df["Maus2_Nase_X"].values
maus2_nase_y = df["Maus2_Nase_Y"].values

# Berechne die Länge des Vektors von (0, 0) zur Nase von Maus 1 und Maus 2
vektor_langen_maus1 = berechne_vektor_laenge(maus1_nase_x, maus1_nase_y)
vektor_langen_maus2 = berechne_vektor_laenge(maus2_nase_x, maus2_nase_y)

# Berechne die Distanz zwischen der Nase von Maus 1 und Maus 2
distanz_maus1_maus2 = berechne_distanz(
    maus1_nase_x, maus1_nase_y, maus2_nase_x, maus2_nase_y
)

# Füge die Ergebnisse in den DataFrame ein (optional)
df["Maus1_Nase_Vektor_Laenge"] = vektor_langen_maus1
df["Maus2_Nase_Vektor_Laenge"] = vektor_langen_maus2
df["Distanz_Maus1_Maus2"] = distanz_maus1_maus2

# Ausgabe der ersten paar berechneten Distanzen und Vektor-Längen
print(
    df[
        [
            "Maus1_Nase_X",
            "Maus1_Nase_Y",
            "Maus1_Nase_Vektor_Laenge",
            "Maus2_Nase_X",
            "Maus2_Nase_Y",
            "Maus2_Nase_Vektor_Laenge",
            "Distanz_Maus1_Maus2",
        ]
    ].head()
)

# %%
import pandas as pd

# Angenommen, du hast bereits die Distanzen zwischen Maus 1 und Maus 2 berechnet
# und diese in der Spalte 'Distanz_Maus1_Maus2' deines DataFrames gespeichert.

# Beispiel: fiktive Frame-Daten
n = len(df)  # Anzahl der Frames
frames = np.arange(1, n + 1)  # Frame-Nummern (falls noch nicht vorhanden)

# Erstelle einen neuen DataFrame mit Frame als Index und der Distanz von Maus 1 zu Maus 2
df_distanz = pd.DataFrame(
    {"Frame": frames, "Distanz_Maus1_Maus2": df["Distanz_Maus1_Maus2"]}
)

# Setze die Frame-Nummern als Index (optional)
df_distanz.set_index("Frame", inplace=True)

# Exportiere den DataFrame in eine Excel-Datei
excel_datei_name = "maus_distanz_frames.xlsx"
df_distanz.to_excel(excel_datei_name)

# Bestätigung, dass die Datei erstellt wurde
print(f"Die Excel-Datei '{excel_datei_name}' wurde erfolgreich erstellt.")

# %%
import os

# Aktuelles Arbeitsverzeichnis anzeigen
print(os.getcwd())

# %%

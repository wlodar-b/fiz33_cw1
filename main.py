"""
Ćw. 1 - Dioda elektroluminescencyjna (LED), Laboratorium Fizyka 3.3.

Opracowanie pomiarów charakterystyk I-V diod LED:
  1) wykresy I-V z niepewnościami u(I), u(U),
  2) potencjał wbudowany V_bi = -b/a z prostej dopasowanej w zakresie dużych napięć,
  3) oporność szeregowa R_s = 1/a,
  4) przerwa energetyczna E_g = hc/lambda,
  5) porównanie z E_g = e*V_bi,
  6) wykresy lnI = f(V) i współczynnik idealności n = q/(k_B T a).

Pliki .tsv: kolumna 1 - napięcie [V], kolumna 2 - prąd [A].
Wyniki: katalog `wykresy/` oraz pliki `wyniki_sprawozdanie.json` i `wyniki_sprawozdanie.txt`.
"""

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------------
# Stałe fizyczne (SI)
# ----------------------------------------------------------------------------
H = 6.62607015e-34  # J*s
C = 299_792_458.0  # m/s
E = 1.602176634e-19  # C
KB = 1.380649e-23  # J/K
T = 300.0  # K

# ----------------------------------------------------------------------------
# Dokładności multimetrów: u = (proc% * |odczyt| + dgt * rozdzielczość) / sqrt(3)
# UWAGA: wpisz wartości z instrukcji do multimetrów używanych na pracowni.
# ----------------------------------------------------------------------------
MULTIMETR_U = {"proc": 0.5, "dgt": 2, "rozdzielczosc": 1e-4}  # woltomierz [V]
MULTIMETR_I = {"proc": 0.8, "dgt": 3, "rozdzielczosc": 1e-6}  # amperomierz [A]

# ----------------------------------------------------------------------------
# Diody: plik, długość fali [nm], kolor wykresu oraz zakresy dopasowań.
#   frac_duze  - do prostej I(V) brane są punkty z I >= frac_duze * I_max
#   zakres_ln  - zakres prądów [A] liniowej części lnI(V)
# ----------------------------------------------------------------------------
DELTA_LAMBDA_NM = 3.0
DIODY = [
    {"nazwa": "niebieska", "plik": "dioda_niebieska.tsv", "lambda_nm": 460,
     "kolor": "tab:blue", "frac_duze": 0.5, "zakres_ln": (3e-5, 2e-4)},
    {"nazwa": "żółto-zielona", "plik": "dioda_zolto-zielona.tsv", "lambda_nm": 560,
     "kolor": "yellowgreen", "frac_duze": 0.5, "zakres_ln": (2e-5, 5e-4)},
    {"nazwa": "pomarańczowa", "plik": "dioda_pomaranczowa.tsv", "lambda_nm": 596,
     "kolor": "tab:orange", "frac_duze": 0.5, "zakres_ln": (2e-5, 5e-4)},
    {"nazwa": "czerwona", "plik": "dioda_czerwona.tsv", "lambda_nm": 650,
     "kolor": "tab:red", "frac_duze": 0.5, "zakres_ln": (2e-5, 5e-4)},
    {"nazwa": "podczerwona", "plik": "dioda_podczerwona.tsv", "lambda_nm": 970,
     "kolor": "darkred", "frac_duze": 0.5, "zakres_ln": (5e-5, 1e-3)},
]

# na wykresach lnI pomijane są prądy poniżej progu (szum rzędu rozdzielczości amperomierza)
PROG_LN_WYKRES = 5e-6  # A

# tabela pomiarów w raporcie: punkty o prądach rozłożonych logarytmicznie od PROG_TABELI do I_max
PUNKTY_TABELI = 20
PROG_TABELI = 1e-5  # A

KATALOG = Path(__file__).resolve().parent
KATALOG_WYKRESOW = KATALOG / "wykresy"


def wczytaj(plik):
    dane = np.loadtxt(KATALOG / plik)
    U, I = dane[:, 0], dane[:, 1]
    kolejnosc = np.argsort(U)
    return U[kolejnosc], I[kolejnosc]


def niepewnosc(odczyt, multimetr):
    dokladnosc = multimetr["proc"] / 100 * np.abs(odczyt) + multimetr["dgt"] * multimetr["rozdzielczosc"]
    return dokladnosc / np.sqrt(3)


def regresja(x, y):
    """Metoda najmniejszych kwadratów y = a*x + b; zwraca a, b, Δa, Δb, R^2."""
    n = len(x)
    xs, ys = x.mean(), y.mean()
    sxx = np.sum((x - xs) ** 2)
    a = np.sum((x - xs) * (y - ys)) / sxx
    b = ys - a * xs
    reszty = y - (a * x + b)
    s2 = np.sum(reszty ** 2) / (n - 2)
    da = np.sqrt(s2 / sxx)
    db = np.sqrt(s2 * (1 / n + xs ** 2 / sxx))
    r2 = 1 - np.sum(reszty ** 2) / np.sum((y - ys) ** 2)
    return a, b, da, db, r2


def analizuj(dioda):
    U, I = wczytaj(dioda["plik"])
    uU = niepewnosc(U, MULTIMETR_U)
    uI = niepewnosc(I, MULTIMETR_I)

    # 2), 3) prosta w zakresie dużych napięć
    maska_duze = I >= dioda["frac_duze"] * I.max()
    a, b, da, db, r2 = regresja(U[maska_duze], I[maska_duze])
    Vbi = -b / a
    uVbi = np.sqrt((b / a ** 2 * da) ** 2 + (db / a) ** 2)
    Rs = 1 / a
    uRs = da / a ** 2

    # 4) E_g z długości fali
    lam = dioda["lambda_nm"] * 1e-9
    u_lam = DELTA_LAMBDA_NM * 1e-9 / np.sqrt(3)
    Eg_lam = H * C / lam / E  # eV
    uEg_lam = H * C / lam ** 2 * u_lam / E

    # 5) E_g = e*V_bi -> w eV liczbowo równe V_bi
    Eg_Vbi, uEg_Vbi = Vbi, uVbi

    # 6) lnI = f(V), współczynnik idealności
    i_min, i_max = dioda["zakres_ln"]
    maska_ln = (I >= i_min) & (I <= i_max)
    a_ln, b_ln, da_ln, db_ln, r2_ln = regresja(U[maska_ln], np.log(I[maska_ln]))
    q_kT = E / (KB * T)
    n_id = q_kT / a_ln
    u_n = q_kT * da_ln / a_ln ** 2
    Is = np.exp(b_ln)
    u_Is = Is * db_ln

    return {
        "U": U, "I": I, "uU": uU, "uI": uI,
        "maska_duze": maska_duze, "maska_ln": maska_ln,
        "a": a, "b": b, "da": da, "db": db, "r2": r2,
        "Vbi": Vbi, "uVbi": uVbi, "Rs": Rs, "uRs": uRs,
        "Eg_lam": Eg_lam, "uEg_lam": uEg_lam, "Eg_Vbi": Eg_Vbi, "uEg_Vbi": uEg_Vbi,
        "a_ln": a_ln, "b_ln": b_ln, "da_ln": da_ln, "db_ln": db_ln, "r2_ln": r2_ln,
        "n": n_id, "u_n": u_n, "Is": Is, "u_Is": u_Is,
    }


def wykres_IV(dioda, w):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(w["U"], w["I"] * 1e3, xerr=w["uU"], yerr=w["uI"] * 1e3, fmt="o", ms=2.5,
                color=dioda["kolor"], ecolor="gray", elinewidth=0.6, capsize=1.5, label="pomiar")
    ax.plot(w["U"][w["maska_duze"]], w["I"][w["maska_duze"]] * 1e3, "o", ms=3, mfc="none",
            color="black", mew=0.6, zorder=3, label="punkty do dopasowania")
    x = np.linspace(w["Vbi"], w["U"].max(), 100)
    ax.plot(x, (w["a"] * x + w["b"]) * 1e3, "k--", lw=1.2,
            label=f"I = aU + b\na = ({w['a']*1e3:.2f} ± {w['da']*1e3:.2f}) mA/V\n"
                  f"b = ({w['b']*1e3:.1f} ± {w['db']*1e3:.1f}) mA,  R² = {w['r2']:.4f}")
    ax.axvline(w["Vbi"], color="k", lw=0.6, ls=":")
    ax.annotate(f"$V_{{bi}}$ = ({w['Vbi']:.3f} ± {w['uVbi']:.3f}) V", (w["Vbi"], 0),
                xytext=(10, 25), textcoords="offset points",
                arrowprops={"arrowstyle": "->", "lw": 0.8})
    ax.set_xlabel("U [V]")
    ax.set_ylabel("I [mA]")
    ax.set_title(f"Charakterystyka I-V – dioda {dioda['nazwa']}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(KATALOG_WYKRESOW / f"IV_{Path(dioda['plik']).stem}.png", dpi=200)
    plt.close(fig)


def wykres_lnI(dioda, w):
    widoczne = w["I"] >= PROG_LN_WYKRES
    U, I = w["U"][widoczne], w["I"][widoczne]
    u_lnI = w["uI"][widoczne] / I
    maska = w["maska_ln"][widoczne]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(U, np.log(I), xerr=w["uU"][widoczne], yerr=u_lnI, fmt="o", ms=2.5,
                color=dioda["kolor"], ecolor="gray", elinewidth=0.6, capsize=1.5, label="pomiar")
    ax.plot(U[maska], np.log(I[maska]), "o", ms=3, mfc="none", color="black", mew=0.6,
            zorder=3, label="punkty do dopasowania")
    x = np.linspace(U[maska].min() - 0.1, U[maska].max() + 0.1, 100)
    ax.plot(x, w["a_ln"] * x + w["b_ln"], "k--", lw=1.2,
            label=f"lnI = aU + b\na = ({w['a_ln']:.2f} ± {w['da_ln']:.2f}) 1/V\n"
                  f"b = ({w['b_ln']:.2f} ± {w['db_ln']:.2f}),  R² = {w['r2_ln']:.4f}\n"
                  f"n = {w['n']:.3f} ± {w['u_n']:.3f}")
    ax.set_ylim(np.log(I).min() - 0.5, np.log(I).max() + 1)
    ax.set_xlabel("U [V]")
    ax.set_ylabel("ln(I / 1 A)")
    ax.set_title(f"Charakterystyka lnI = f(U) – dioda {dioda['nazwa']}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(KATALOG_WYKRESOW / f"lnI_{Path(dioda['plik']).stem}.png", dpi=200)
    plt.close(fig)


def wykresy_zbiorcze(wyniki):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for dioda, w in wyniki:
        ax.plot(w["U"], w["I"] * 1e3, "o", ms=2, color=dioda["kolor"], label=dioda["nazwa"])
    ax.set_xlabel("U [V]")
    ax.set_ylabel("I [mA]")
    ax.set_title("Charakterystyki I-V badanych diod LED")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(KATALOG_WYKRESOW / "IV_wszystkie.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for dioda, w in wyniki:
        widoczne = w["I"] >= PROG_LN_WYKRES
        ax.plot(w["U"][widoczne], np.log(w["I"][widoczne]), "o", ms=2, color=dioda["kolor"],
                label=dioda["nazwa"])
    ax.set_xlabel("U [V]")
    ax.set_ylabel("ln(I / 1 A)")
    ax.set_title("Charakterystyki lnI = f(U) badanych diod LED")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(KATALOG_WYKRESOW / "lnI_wszystkie.png", dpi=200)
    plt.close(fig)

    nazwy = [d["nazwa"] for d, _ in wyniki]
    x = np.arange(len(nazwy))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(x - 0.1, [w["Eg_lam"] for _, w in wyniki], yerr=[w["uEg_lam"] for _, w in wyniki],
                fmt="o", capsize=4, label=r"$E_g = hc/\lambda$")
    ax.errorbar(x + 0.1, [w["Eg_Vbi"] for _, w in wyniki], yerr=[w["uEg_Vbi"] for _, w in wyniki],
                fmt="s", capsize=4, label=r"$E_g = e\,V_{bi}$")
    ax.set_xticks(x, nazwy)
    ax.set_ylabel(r"$E_g$ [eV]")
    ax.set_title("Porównanie przerw energetycznych")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(KATALOG_WYKRESOW / "Eg_porownanie.png", dpi=200)
    plt.close(fig)


def zaokraglij(val, unc, cyfry=2):
    """Niepewność do `cyfry` cyfr znaczących, wartość do tego samego miejsca dziesiętnego."""
    if unc is None or not np.isfinite(unc) or unc <= 0:
        return {"val": f"{val:.6g}", "unc": None}
    rzad = math.floor(math.log10(unc))
    if rzad < -4 or rzad > 5:
        wykladnik = math.floor(math.log10(abs(val))) if val != 0 else rzad
        val, unc = val / 10 ** wykladnik, unc / 10 ** wykladnik
        miejsca = max(cyfry - 1 - math.floor(math.log10(unc)), 0)
        return {"val": f"{val:.{miejsca}f}e{wykladnik}", "unc": f"{unc:.{miejsca}f}e{wykladnik}"}
    miejsca = cyfry - 1 - rzad
    if miejsca > 0:
        return {"val": f"{val:.{miejsca}f}", "unc": f"{unc:.{miejsca}f}"}
    return {"val": f"{round(val, miejsca):.0f}", "unc": f"{round(unc, miejsca):.0f}"}


def pm(wartosc):
    return f"({wartosc['val']} +/- {wartosc['unc']})"


def wybrane_punkty(w):
    """Indeksy punktów o prądach rozłożonych logarytmicznie od PROG_TABELI do I_max."""
    lnI = np.log(np.clip(w["I"], 1e-12, None))
    cele = np.log(np.geomspace(PROG_TABELI, w["I"].max(), PUNKTY_TABELI))
    return sorted({int(np.argmin(np.abs(lnI - c))) for c in cele})


def zbuduj_raport(wyniki):
    u_lambda = DELTA_LAMBDA_NM / np.sqrt(3)
    raport = {
        "cwiczenie": "Ćw. 1 - Dioda elektroluminescencyjna (LED), Laboratorium Fizyka 3.3",
        "parametry_obliczen": {
            "T_K": T,
            "q_kBT_1_per_V": round(E / (KB * T), 3),
            "delta_lambda_nm": DELTA_LAMBDA_NM,
            "u_lambda_nm": round(u_lambda, 2),
            "multimetr_U": {**MULTIMETR_U, "jednostka": "V"},
            "multimetr_I": {**MULTIMETR_I, "jednostka": "A"},
            "wzor_niepewnosci_miernika": "u = (proc% * |odczyt| + dgt * rozdzielczosc) / sqrt(3)",
        },
        "parametry_fizyczne": {},
        "wybrane_pomiary": {},
    }

    for dioda, w in wyniki:
        klucz = Path(dioda["plik"]).stem
        U, I = w["U"], w["I"]
        md, ml = w["maska_duze"], w["maska_ln"]
        i_min, i_max = dioda["zakres_ln"]
        roznica = w["Eg_Vbi"] - w["Eg_lam"]

        raport["parametry_fizyczne"][klucz] = {
            "nazwa": dioda["nazwa"],
            "lambda_nm": dioda["lambda_nm"],
            "liczba_punktow_pomiarowych": int(len(U)),
            "I_max_mA": round(float(I.max()) * 1e3, 3),
            "prosta_I_od_U": {
                "kryterium": f"I >= {dioda['frac_duze']:.0%} I_max",
                "zakres_U_V": [round(float(U[md].min()), 4), round(float(U[md].max()), 4)],
                "zakres_I_mA": [round(float(I[md].min()) * 1e3, 3), round(float(I[md].max()) * 1e3, 3)],
                "liczba_punktow": int(md.sum()),
                "a_mA_per_V": zaokraglij(w["a"] * 1e3, w["da"] * 1e3),
                "b_mA": zaokraglij(w["b"] * 1e3, w["db"] * 1e3),
                "R2": round(w["r2"], 5),
            },
            "Vbi_V": zaokraglij(w["Vbi"], w["uVbi"]),
            "Rs_ohm": zaokraglij(w["Rs"], w["uRs"]),
            "Eg_lambda_eV": zaokraglij(w["Eg_lam"], w["uEg_lam"]),
            "Eg_Vbi_eV": zaokraglij(w["Eg_Vbi"], w["uEg_Vbi"]),
            "roznica_Eg_Vbi_minus_Eg_lambda_eV": round(roznica, 3),
            "roznica_wzgledna_procent": round(roznica / w["Eg_lam"] * 100, 1),
            "prosta_lnI_od_U": {
                "zakres_I_A": [i_min, i_max],
                "zakres_U_V": [round(float(U[ml].min()), 4), round(float(U[ml].max()), 4)],
                "liczba_punktow": int(ml.sum()),
                "a_1_per_V": zaokraglij(w["a_ln"], w["da_ln"]),
                "b": zaokraglij(w["b_ln"], w["db_ln"]),
                "R2": round(w["r2_ln"], 5),
            },
            "n": zaokraglij(w["n"], w["u_n"]),
            "Is_A": zaokraglij(w["Is"], w["u_Is"]),
        }

        raport["wybrane_pomiary"][klucz] = [
            {
                "lp": lp,
                "U_V": zaokraglij(U[i], w["uU"][i]),
                "I_mA": zaokraglij(I[i] * 1e3, w["uI"][i] * 1e3),
                "ln_I": zaokraglij(np.log(I[i]), w["uI"][i] / I[i]),
            }
            for lp, i in enumerate(wybrane_punkty(w), start=1)
        ]
    return raport


def raport_txt(raport):
    linia = "=" * 80
    p = raport["parametry_obliczen"]
    mu, mi = p["multimetr_U"], p["multimetr_I"]
    out = [
        linia,
        "PODSUMOWANIE WYNIKÓW DO SPRAWOZDANIA - ĆWICZENIE 1",
        "Dioda elektroluminescencyjna (LED) - charakterystyki I-V diod LED",
        linia,
        "",
        "PARAMETRY OBLICZEŃ",
        f"   Temperatura T             : {p['T_K']} K,  q/(k_B T) = {p['q_kBT_1_per_V']} 1/V",
        f"   Niepewność długości fali  : Δλ = {p['delta_lambda_nm']} nm,  u(λ) = Δλ/sqrt(3) = {p['u_lambda_nm']} nm",
        f"   Woltomierz                : ±({mu['proc']}% rdg + {mu['dgt']} dgt), rozdzielczość {mu['rozdzielczosc']} V",
        f"   Amperomierz               : ±({mi['proc']}% rdg + {mi['dgt']} dgt), rozdzielczość {mi['rozdzielczosc']} A",
        f"   Niepewność miernika       : {p['wzor_niepewnosci_miernika']}",
        "",
    ]

    for nr, (klucz, d) in enumerate(raport["parametry_fizyczne"].items(), start=1):
        pr, pl = d["prosta_I_od_U"], d["prosta_lnI_od_U"]
        out += [
            f"{nr}. DIODA {d['nazwa'].upper()} ({klucz}.tsv) - λ = {d['lambda_nm']} nm, "
            f"N = {d['liczba_punktow_pomiarowych']} pkt, I_max = {d['I_max_mA']} mA",
            f"   Prosta I = aU + b, {pr['kryterium']}: U = {pr['zakres_U_V'][0]} - {pr['zakres_U_V'][1]} V,"
            f" I = {pr['zakres_I_mA'][0]} - {pr['zakres_I_mA'][1]} mA (N = {pr['liczba_punktow']})",
            f"   Współczynnik kierunkowy a : {pm(pr['a_mA_per_V'])} mA/V",
            f"   Wyraz wolny b             : {pm(pr['b_mA'])} mA",
            f"   Współczynnik determinacji : R^2 = {pr['R2']}",
            f"   Potencjał wbudowany V_bi  : {pm(d['Vbi_V'])} V",
            f"   Oporność szeregowa R_s    : {pm(d['Rs_ohm'])} Ohm",
            f"   Przerwa Eg = hc/λ         : {pm(d['Eg_lambda_eV'])} eV",
            f"   Przerwa Eg = e*V_bi       : {pm(d['Eg_Vbi_eV'])} eV",
            f"   Różnica Eg(V_bi) - Eg(λ)  : {d['roznica_Eg_Vbi_minus_Eg_lambda_eV']:+.3f} eV"
            f" ({d['roznica_wzgledna_procent']:+.1f} %)",
            f"   Prosta lnI = aU + b, I = {pl['zakres_I_A'][0]:g} - {pl['zakres_I_A'][1]:g} A:"
            f" U = {pl['zakres_U_V'][0]} - {pl['zakres_U_V'][1]} V (N = {pl['liczba_punktow']})",
            f"   Współczynnik kierunkowy a : {pm(pl['a_1_per_V'])} 1/V",
            f"   Wyraz wolny b = ln(I_s)   : {pm(pl['b'])} [-]",
            f"   Współczynnik determinacji : R^2 = {pl['R2']}",
            f"   Współczynnik idealności n : {pm(d['n'])}",
            f"   Prąd nasycenia I_s        : {pm(d['Is_A'])} A",
            "",
        ]

    out += [
        linia,
        "TABELA ZBIORCZA",
        linia,
        f"{'Dioda':<14} | {'λ [nm]':>6} | {'V_bi [V]':>20} | {'R_s [Ohm]':>20} | {'Eg=hc/λ [eV]':>20} |"
        f" {'Eg=eV_bi [eV]':>20} | {'n':>20}",
        "-" * 140,
    ]
    for d in raport["parametry_fizyczne"].values():
        out.append(
            f"{d['nazwa']:<14} | {d['lambda_nm']:>6} | {pm(d['Vbi_V']):>20} | {pm(d['Rs_ohm']):>20} |"
            f" {pm(d['Eg_lambda_eV']):>20} | {pm(d['Eg_Vbi_eV']):>20} | {pm(d['n']):>20}"
        )

    out += [
        "",
        linia,
        f"WYBRANE PUNKTY POMIAROWE (ok. {PUNKTY_TABELI} na diodę, prądy rozłożone logarytmicznie"
        f" od {PROG_TABELI:g} A do I_max; niepewności do 2 cyfr znaczących)",
        linia,
    ]
    for klucz, punkty in raport["wybrane_pomiary"].items():
        out += [
            "",
            f"Dioda {raport['parametry_fizyczne'][klucz]['nazwa']}",
            f"{'Lp':>2} | {'U +/- u(U) [V]':>22} | {'I +/- u(I) [mA]':>24} | {'ln(I/1A) +/- u':>22}",
            "-" * 80,
        ]
        for pkt in punkty:
            out.append(
                f"{pkt['lp']:>2} | {pkt['U_V']['val'] + ' +/- ' + pkt['U_V']['unc']:>22} |"
                f" {pkt['I_mA']['val'] + ' +/- ' + pkt['I_mA']['unc']:>24} |"
                f" {pkt['ln_I']['val'] + ' +/- ' + pkt['ln_I']['unc']:>22}"
            )
    return "\n".join(out) + "\n"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    KATALOG_WYKRESOW.mkdir(exist_ok=True)
    wyniki = []
    for dioda in DIODY:
        w = analizuj(dioda)
        wykres_IV(dioda, w)
        wykres_lnI(dioda, w)
        wyniki.append((dioda, w))
    wykresy_zbiorcze(wyniki)

    raport = zbuduj_raport(wyniki)
    tekst = raport_txt(raport)
    plik_json = KATALOG / "wyniki_sprawozdanie.json"
    plik_txt = KATALOG / "wyniki_sprawozdanie.txt"
    plik_json.write_text(json.dumps(raport, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plik_txt.write_text(tekst, encoding="utf-8")

    print(tekst)
    print(f"Wykresy zapisano w: {KATALOG_WYKRESOW}")
    print(f"Wyniki zapisano w:  {plik_json.name}, {plik_txt.name}")


if __name__ == "__main__":
    main()
